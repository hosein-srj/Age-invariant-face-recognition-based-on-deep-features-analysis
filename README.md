# Age invariant face recognition based on deep features analysis
paper: https://link.springer.com/article/10.1007/s11760-020-01635-1

Pre-proccessing:

![image](https://user-images.githubusercontent.com/54143711/135315895-e9eea182-4c0f-4063-a877-cb16eed3f03a.png)

![image](https://user-images.githubusercontent.com/54143711/135315985-781b475a-55e3-4c10-ac95-d40e6d82121e.png)

feature extraction (deelte last softmax layer):

![image](https://user-images.githubusercontent.com/54143711/135316151-6f2320f5-1c19-443b-be1b-f9d89adc031f.png)

feature fusion Discriminant correlation analysis (DCA):

![image](https://user-images.githubusercontent.com/54143711/135316289-98de7aff-78d5-4638-9860-9a5ea334f23a.png)

Classification with KNN with K=1 and SVM

article accuracy:

![image](https://user-images.githubusercontent.com/54143711/137171356-b7f1fd4d-1c6b-44e5-82aa-b45bce3971ed.png)

my implepemtation: <br>
dataset FGNET KNN with K=5: 78.56% <br>
dataset FGNET SVM: 89.87% <br>
