# Generated by Django 4.0.5 on 2022-06-30 17:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='name',
            field=models.CharField(default='new_photo.png', max_length=255),
        ),
    ]