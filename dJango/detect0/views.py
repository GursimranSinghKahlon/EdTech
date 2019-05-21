from django.shortcuts import render
from django.http import HttpResponse
from .models import Product


from . import nlpo4

import os

static_path = os.path.join(os.getcwd(),"detect0","static")
print(static_path)

def index(request):
    return render(request, 'index3.html')
 
 
def findAns(request):
	query = request.POST.get('ques',None)
	count = int(request.POST.get('count',1))
	print(query)
	result = nlpo4.getAns(query,
           top=count,
           data_file = os.path.join(static_path,'ques_ans4.json'),
           saved_model = os.path.join(static_path,'finalizedLR_model.sav') )

	print(result)
	
	return render(request, 'findAns.html',{'question': result})

  

