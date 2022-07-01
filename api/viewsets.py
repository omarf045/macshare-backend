from rest_framework import viewsets

from rest_framework.response import Response
from rest_framework.decorators import action

from .models import Photo
from .serializers import PhotoSerializer
from rest_framework.authtoken.models import Token


class PhotoViewSet(viewsets.ModelViewSet):

    #queryset = Photo.objects.all()
    serializer_class = PhotoSerializer

    """@action(detail=True, methods=['get'])
    def user_photos(self, request):

        str_token = request.META.get('HTTP_AUTHORIZATION')
        username = Token.objects.get(key=str_token).user.username
        print(username)

        photos = Photo.objects.filter(username=username).distinct()
        schedule_json = PhotoSerializer(photos, many=True)
        return Response(schedule_json.data)"""

    def get_queryset(self):
        user = self.request.user
        print(user)
        return Photo.objects.filter(username=user)
