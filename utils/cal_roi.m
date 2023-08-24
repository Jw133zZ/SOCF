function score = cal_roi(pos_hc,pos,target_sz)
%���㵱ǰλ�õ�����λ��
xy = pos_hc([2,1])-(target_sz([2,1]))/2;
X_a1 = xy(1);Y_a1 = xy(2);
xy2 = xy + target_sz([2,1]);
X_a2 = xy2(1); Y_a2 = xy2(2);
%����ǰһ�������λ��
xy_old = pos([2,1])-(target_sz([2,1]))/2;
X_b1 = xy_old(1);Y_b1 = xy_old(2);
xy_old2 = xy_old + target_sz([2,1]);
X_b2 = xy_old2(1); Y_b2 = xy_old2(2);
%���㽻����
x1 = max(X_a1,X_b1); x2 = min(X_a2,X_b2);
y1 = max(Y_b1,Y_a1); y2 = min(Y_b2,Y_a2);
intersection = max(x2-x1,0)* max(y2-y1,0);
S_a = (X_a2-X_a1)*(Y_a2-Y_a1);
union = 2*S_a - intersection;
score = intersection/union;
end

