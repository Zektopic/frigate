import peewee as pw




def migrate(migrator, database, fake=False, **kwargs):
    migrator.change_fields(
        'event', sub_label=pw.CharField(max_length=100, null=True))


def rollback(migrator, database, fake=False, **kwargs):
    migrator.change_fields(
        'event', sub_label=pw.CharField(max_length=20, null=True))
