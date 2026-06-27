from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    task_order INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    description TEXT NOT NULL,
    input_files TEXT NOT NULL DEFAULT '',
    output_file TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT 'coding',
    review_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tasks_session_status_order
    ON tasks (session_id, status, task_order);
"""


@dataclass
class TaskRecord:
    id: int
    session_id: str
    task_order: int
    status: str
    description: str
    input_files: str
    output_file: str
    context: str
    task_type: str
    review_count: int


def _row_to_task(row: tuple) -> TaskRecord:
    return TaskRecord(
        id=row[0],
        session_id=row[1],
        task_order=row[2],
        status=row[3],
        description=row[4],
        input_files=row[5],
        output_file=row[6],
        context=row[7],
        task_type=row[8],
        review_count=row[9],
    )


class TaskQueue:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(_SCHEMA)

    def add_task(
        self,
        session_id: str,
        description: str,
        task_type: str,
        *,
        input_files: str = '',
        output_file: str = '',
        context: str = '',
        initial_review_count: int = 0,
    ) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                'SELECT COALESCE(MAX(task_order), 0) FROM tasks WHERE session_id = ?',
                (session_id,),
            ).fetchone()
            next_order = row[0] + 1
            cursor = conn.execute(
                'INSERT INTO tasks (session_id, task_order, description, task_type, '
                'input_files, output_file, context, review_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (session_id, next_order, description, task_type, input_files, output_file, context, initial_review_count),
            )
            conn.commit()
            return cursor.lastrowid

    def add_correction(
        self,
        session_id: str,
        after_task_order: int,
        description: str,
        task_type: str,
        *,
        input_files: str = '',
        output_file: str = '',
        context: str = '',
        initial_review_count: int = 0,
    ) -> int:
        """Insert a correction task immediately after the failed task.

        Shifts all later pending tasks up by one position so the correction
        runs before any subsequent planned tasks.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE tasks SET task_order = task_order + 1 "
                "WHERE session_id = ? AND task_order > ? AND status = 'pending'",
                (session_id, after_task_order),
            )
            next_order = after_task_order + 1
            cursor = conn.execute(
                'INSERT INTO tasks (session_id, task_order, description, task_type, '
                'input_files, output_file, context, review_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (session_id, next_order, description, task_type,
                 input_files, output_file, context, initial_review_count),
            )
            conn.commit()
            return cursor.lastrowid

    def block_remaining(self, session_id: str, after_task_order: int) -> int:
        """Mark all pending tasks after `after_task_order` as blocked. Returns count blocked."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "UPDATE tasks SET status = 'blocked' "
                "WHERE session_id = ? AND status = 'pending' AND task_order > ?",
                (session_id, after_task_order),
            )
            conn.commit()
            return cursor.rowcount

    def get_blocked(self, session_id: str) -> list[TaskRecord]:
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                'SELECT id, session_id, task_order, status, description, input_files, '
                'output_file, context, task_type, review_count '
                "FROM tasks WHERE session_id = ? AND status = 'blocked' ORDER BY task_order",
                (session_id,),
            ).fetchall()
            return [_row_to_task(row) for row in rows]

    def get_next_pending(self, session_id: str) -> TaskRecord | None:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                'SELECT id, session_id, task_order, status, description, input_files, '
                'output_file, context, task_type, review_count '
                'FROM tasks WHERE session_id = ? AND status = ? '
                'ORDER BY task_order ASC LIMIT 1',
                (session_id, 'pending'),
            ).fetchone()
            return _row_to_task(row) if row else None

    def update_status(self, task_id: int, status: str) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('UPDATE tasks SET status = ? WHERE id = ?', (status, task_id))
            conn.commit()

    def reset_in_progress(self, session_id: str) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(
                "UPDATE tasks SET status = 'pending' WHERE session_id = ? AND status = 'in_progress'",
                (session_id,),
            )
            conn.commit()
            return cursor.rowcount

    def count_pending(self, session_id: str) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).fetchone()
            return row[0] if row else 0

    def count_session_tasks(self, session_id: str) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                'SELECT COUNT(*) FROM tasks WHERE session_id = ?', (session_id,)
            ).fetchone()
            return row[0] if row else 0

    def find_by_output_file(self, session_id: str, output_file: str) -> int | None:
        """Return the task ID of any existing task writing to output_file, or None."""
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                'SELECT id FROM tasks WHERE session_id = ? AND output_file = ? LIMIT 1',
                (session_id, output_file),
            ).fetchone()
            return row[0] if row else None

    def update_context(self, task_id: int, context: str) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('UPDATE tasks SET context = ? WHERE id = ?', (context, task_id))
            conn.commit()

    def increment_review_count(self, task_id: int) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                'UPDATE tasks SET review_count = review_count + 1 WHERE id = ?', (task_id,)
            )
            conn.commit()
            row = conn.execute(
                'SELECT review_count FROM tasks WHERE id = ?', (task_id,)
            ).fetchone()
            return row[0] if row else 0
