
fileID = fopen('data.txt','r');
Flatten = [];
FC7 = [];
FC6 = [];
labels = [];
for i=1:1002
    l_str = fgets(fileID);
    Flaten_str = fgets(fileID);
    FC7_str = fgets(fileID);
    FC6_str = fgets(fileID);
    l = str2double(l_str);
    labels = [ labels , l];
    Flatten = [Flatten ,str2double(split(Flaten_str,' '))];
    FC7 = [FC7 ,str2double(split(FC7_str,' '))];
    FC6 = [FC6 ,str2double(split(FC6_str,' '))];
    if mod(i,50)==0
       datetime('now')
    end
end

[Ax1,Ay1,Xs1,Ys1] = dcaFuse(Flatten,FC6,labels);

[Ax2,Ay2,Xs2,Ys2] = dcaFuse(FC7,FC6,labels);

level2_X = Xs1 + Ys1;
level2_Y = Xs2 + Ys2;

[Ax_out,Ay_out,Xs_out,Ys_out] = dcaFuse(level2_X,level2_Y,labels);

final_output = Xs_out + Ys_out;

fid = fopen('E:\features.txt','w');
t = final_output';
for i=1:1002
   
   for j=1:81
      fprintf(fid,"%f ",t(i,j));
   end
   fprintf(fid,"\n");
end

fclose(fid);