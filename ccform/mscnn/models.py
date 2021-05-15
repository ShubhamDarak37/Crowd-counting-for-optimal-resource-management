from django.db import models

# Create your models here.
class countPrediction(models.Model): 
    CrowdCount = models.IntegerField(default=0) 
    CDate = models.DateField(auto_now_add=False,default = '2001-01-01',null = True)
    CTime = models.TimeField(auto_now_add=False,default = '20:00',null =True)