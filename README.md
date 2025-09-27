Mindyolo Prune And Performance Testing
====================

About Prune
--------

   well，this may be the first code about mindyolo prune.Prune code maybe a lot about Pytorch,but in mindspore,there's none.I'm just a student,and make this just for my study.So ,there maybe a lot error,and it's performance may not satisfied you.But if want to give some suggestions, please refer to  `weiyuhao.nwpu.edu.cn@mail.nwpu.edu.cn `.<br>
   Here is the details about how to prune yourself mindyolo model,but only yolov8.Because every version of the model ,its network structure isn't same.But you can do some fitness to your model.Other wise ,it's just a very easy Channel pruning<br>
   ```
   git clone git@github.com:yuzhou137/ict-2025.git
   cd mindyolo-master
   pip install -r requirements.txt
   python mindyolo_prue.py --weight \<your_ckpt_file> -c \<your_config_file> --pr \<your_prune_rate>
   ```

   
   `Then you successfully prune your model!`<br>

## some notice about prune
   Well,in case of that the BN layers is not evenly distributed in the model ,so may the real pr rate isn' same to what you set.It' OK.Maybe there`s someone who will write better code than me .And then ,you can try them.<br>
   By the way,I set the default save path is current directory ,and the model name is  mindspore_yolov8s.ckpt.You can also change it if you want.<br>


About Performance Testing
-------

  This part is about how to test your model, and give the visualization performance results.Here is how you use it.<br>

  ```
  python val.py -c \<your_config_file> --weight \<your_ckpt_file>
  
  ```
## Some notice about testing
  When I wrote this code,there is a error that I can't avoid.So if you want to use it ,you have to change the line 225 to the path where your dataset laid.<br>
  And your dataset structure must be like
  ```
  dataset
  --val
  ----images
  ----labels
  ```
  And if your images is png file,you also need to change the line 260,change the jpg to png.<br>
  



`That's all,thanks for supporting`

  well, after some attempts,I found that the purned model have channels mismatch problems.Well,I did some work to correct it,but when I tried it ,the effort of model is really bad,I guess it must be the model need fintune,but I`ve already finish my work .Unless it`s needed,I won`t update my code and complete the whole processing of prune model
  

