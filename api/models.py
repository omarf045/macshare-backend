from multiprocessing import shared_memory
from unicodedata import category
from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Photo(models.Model):
    id = models.AutoField(primary_key=True)
    category = models.CharField(max_length=16, null=True)
    username = models.ForeignKey(
        User, max_length=16, on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=255, default="new_photo.png")
    dimensions = models.CharField(max_length=32)
    created = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)
    size = models.CharField(max_length=32)
    photo = models.ImageField(upload_to='images')
    shared_with = models.JSONField(null=True)
