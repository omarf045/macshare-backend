from django.urls import path
from .views import UserDetailAPI, RegisterUserAPIView, UserPhotoListAPI, ProtectedPhotoAPI, PhotoVerifyAPI
urlpatterns = [
    path("get-details", UserDetailAPI.as_view()),
    path('register', RegisterUserAPIView.as_view()),
    path('photos', UserPhotoListAPI.as_view()),
    path('photo-protect', ProtectedPhotoAPI.as_view()),
    path('photo-verify', PhotoVerifyAPI.as_view()),
]
