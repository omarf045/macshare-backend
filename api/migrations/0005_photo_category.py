# Generated by Django 4.0.5 on 2022-07-01 17:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_alter_photo_username'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='category',
            field=models.CharField(max_length=16, null=True),
        ),
    ]
