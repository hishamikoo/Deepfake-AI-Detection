from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.index, name='index'),
    path('upload_audio/', views.upload_audio, name='upload_audio'),
    
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)