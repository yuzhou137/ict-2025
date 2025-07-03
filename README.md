Compared to Mindyolo, our project has the following improvements
1.change the logical of the draw_results function in mindyolo.utils.utils ,well compared to transmitting a image_path ,we choose to transfer a numpy array of images，and it makes it more efficient and convenient
2.Added val. py file to visualize a model in the form of a confusion matrix
3.Added demo/predict2.py file to enable the model to handle video streams


Usage :
1.for draw_result function,you just need to transfer a numpy array of a image to it ,rather than transfer a directly path of image ,it decilne the cost of CPU ,and accelerate the processing speed of the model
2.for val.py , cause of the limited capacity of myself ,it has a little work to do before you use.
First,change the line 201 -- nc to you nc
change the line 255 -- dataset path to your path
3.for the predict2.py ,also because of my self ,you have to change the line 425 -- change the args.data.names to yours 


Well ,there maybe some others to change ,but that`s what I can remeber .I`m just a student ,so sometimes I may be lazy ,just for convenient for our project ,yes ,I don`t have enough time to make the code sufficient versatility.But I have to say ,compared to the yolo ,there`s still a lot to do to run over it.It was so simple and crude that I was structed when I first contact and use it .Hopes that our effort can do our bit
