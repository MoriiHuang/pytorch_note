from typing import List
import numpy as np
# https://zhuanlan.zhihu.com/p/37777074
class Value_error(Exception):
    def __init__(self,msg) -> None:
        self.msg=msg
    def __str__(self) -> str:
        return self.msg
matrix=List[List[int]]
class PCA:
    def __init__(self,data:matrix,k:int) -> None:
        self.data=data
        self.k=k
        self.samples=len(data)
        self.features=len(data[0])
        self.pca_data=[]
        self.scatter_mat=[]
    def preprocessing(self):
        mean_list=[]
        norm_list=[]
        sum_=[0]*self.features
        for i in range(self.samples):
            for j in range(self.features):
                sum_[j]+=self.data[i][j]
        for i in sum_:
            mean_list.append(i/self.samples)
        for i in range(self.samples):
            tmp=[]
            for j in range(self.features):
                tmp.append(self.data[i][j]-mean_list[j])
            norm_list.append(tmp)
        self.mean_=np.array(mean_list)
        self.norm_=np.array(norm_list)
    def scatter_matrix(self):
        # print(np.transpose(self.norm_).shape,self.norm_.shape)
        self.scatter_mat=np.dot((np.transpose(self.norm_)),self.norm_)
    def eig(self):
        eig_val,eig_vec=np.linalg.eig(self.scatter_mat)
        self.eig_val=eig_val
        self.eig_vec=eig_vec
        eig_pairs=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(self.features)]
        eig_pairs.sort(key=lambda k:k[0],reverse=True)
        self.eig_pairs=eig_pairs
    def pca(self):
        self.preprocessing()
        try :
            if self.k>min(self.features,self.samples):
                raise Value_error("n_components=3 must be between 0 and min(n_samples, n_features)=2 with svd_solver='full'")
            else:
                self.scatter_matrix()
                self.eig()
                features=np.array([ele[1] for ele in self.eig_pairs[:self.k]])
                self.pca_data=[np.dot(self.norm_,np.transpose(features))]
                return self.pca_data
        except Value_error as e:
            print("{}".format(e))
    def SVD(self):# X=U\SigmaV^{T},U是XX^{T}的特征向量按列组成的矩阵，V是X^{T}X的特征向量按列组成的矩阵 #压缩列用V，Y=X_normV(前k行)^{T},压缩行用U，Y=U(前k行)*X_norm
        self.preprocessing()
        U,s,V=np.linalg.svd(self.norm_)
        tmp=V[:self.k]
        self.pca_data=np.dot(self.norm_,tmp.T)
        return self.pca_data
x=[[1,2,3,4,5],[6,7,8,9,10],[3,2,1,233,2]]
print(np.array(x).shape)
p=PCA(x,2)
s1=p.pca()
s2=p.SVD()
print(s1)
print(s2)
