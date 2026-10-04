"""Peewee migrations -- 036_add_has_clip_to_event.py."""

import peewee as pw

SQL = pw.SQL


def migrate(migrator, database, fake=False, **kwargs):
    try:
        migrator.add_fields("event", has_clip=pw.BooleanField(default=True))
    except Exception:
        pass

    try:
        migrator.add_fields("event", has_snapshot=pw.BooleanField(default=True))
    except Exception:
        pass


def rollback(migrator, database, fake=False, **kwargs):
    pass
