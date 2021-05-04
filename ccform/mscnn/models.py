from django.db import models

# Create your models here.
class countPrediction(models.Model): 
    CrowCount = models.IntegerField() 
    InputImage = models.ImageField(upload_to='images/')
    