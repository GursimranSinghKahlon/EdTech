# EdTech

Training data:
https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
@ https://gluebenchmark.com/tasks


Use predict.ipynb to predict answers.

getAns(query,
           top=1,
           data_file = 'ques_ans4.json',
           saved_model = 'finalizedLR_model.sav' )



For django Project:

HomePage(dJango/detect0/templates/index3.html):
http://127.0.0.1:8000/detect0/

![alt text](https://github.com/GursimranSinghKahlon/EdTech/blob/master/dJango/detect0/Screenshots/home.png)


Output(dJango/detect0/templates/findAns.html):
http://127.0.0.1:8000/detect0/findAns

![alt text](https://github.com/GursimranSinghKahlon/EdTech/tree/master/dJango/detect0/Screenshots/result.png)


Configuration for model, question file can be chaged here:
dJango/detect0/views.py

![alt text](https://github.com/GursimranSinghKahlon/EdTech/tree/master/dJango/detect0/Screenshots/settings.png)
