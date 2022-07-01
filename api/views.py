# Create your views here.
import json
from rest_framework.permissions import AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework import authentication, permissions

from PIL import Image
from rest_framework.renderers import JSONRenderer
from django.http import FileResponse, HttpResponse, HttpResponseNotFound
from .utils import get_protected_image, verify_image
import os
from rest_framework.decorators import parser_classes
from rest_framework.parsers import MultiPartParser
from django.conf import settings

from .models import Photo
from .serializers import UserSerializer, RegisterSerializer, PhotoSerializer
from django.contrib.auth.models import User
from rest_framework.authentication import TokenAuthentication
from rest_framework import generics
from rest_framework.authtoken.models import Token
# Class based view to Get User Details using Token Authentication


class UserDetailAPI(APIView):
    authentication_classes = (TokenAuthentication,)
    permission_classes = (AllowAny,)

    def get(self, request, *args, **kwargs):
        user = User.objects.get(id=request.user.id)
        serializer = UserSerializer(user)
        return Response(serializer.data)


# Class based view to register user


class RegisterUserAPIView(generics.CreateAPIView):
    permission_classes = (AllowAny,)
    serializer_class = RegisterSerializer


class UserPhotoListAPI(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request, *args, **kwargs):

        user = request.user.id

        print("GET: ")
        # print(user)

        photos = Photo.objects.filter(username=user)

        serializer = PhotoSerializer(photos, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):

        print("POST")
        user_id = request.user.id
        print(user_id)

        data = request.data

        data['username'] = user_id

        serializer = PhotoSerializer(data=data)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ProtectedPhotoAPI(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = (permissions.IsAuthenticated,)

    def get(self, request, *args, **kwargs):
        user = request.user.id

        photo = Photo.objects.filter(username=user).filter(
            id=request.data['id_photo'])

        id_shared_with = request.data['id_shared_with']

        serializer = PhotoSerializer(photo, many=True)

        json_bytes = JSONRenderer().render(serializer.data)

        data_string = json_bytes.decode('utf-8')
        photo_json = json.loads(data_string)

        path = "." + photo_json[0]['photo']

        photo_name = str(photo_json[0]['id']) + "_" + photo_json[0]['name']

        get_protected_image(path, id_shared_with, photo_name)

        file_location = './media/output/' + photo_name + '.png'

        try:
            f = open(file_location, 'rb')
            file_data = f.read()

            # sending response
            response = HttpResponse(
                file_data, content_type='image/png')
            response['Content-Disposition'] = 'attachment; filename="' + \
                photo_name + '.png"'

        except IOError:
            # handle file not exist case here
            response = HttpResponseNotFound('<h1>File not exist</h1>')

        f.close()
        os.remove(file_location)

        return response


class PhotoVerifyAPI(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = (permissions.IsAuthenticated,)
    parser_classes = (MultiPartParser,)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES['file']

        path = os.path.join(settings.MEDIA_ROOT + '/input/', file_obj.name)

        with open(path, 'wb') as infile:
            str_repr = file_obj.read()
            infile.write(str_repr)
            infile.close()

        index = verify_image(path)

        data = {'index': index, 'rel_with': (index + 1)}

        os.remove(path)

        return Response(data)
