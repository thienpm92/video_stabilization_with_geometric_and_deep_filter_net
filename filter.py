import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as linalg
from scipy.sparse.linalg import inv
from functools import partial


def first_derivative_matrix(n):
    #  version 1
    # dif_now = np.diag(np.ones(n))
    # dif_pre_one = np.ones(n-1)*(-1)
    # dif_pre = np.diag(dif_pre_one,k=-1)
    # D = dif_now + dif_pre
    # D = D[1:,:]
    #  version 2
    data = np.repeat([[-1],[1]],n,axis=1)
    D = sparse.dia_matrix((data,[0,1]),shape=(n-1,n))
    return D

def second_derivative_matrix(n):
    #  version 1
    # dif_central = np.diag(np.ones(n)*-2)
    # one_vec = np.ones(n-1)
    # dif_left = np.diag(one_vec,k=-1)
    # dif_right = np.diag(one_vec,k=1)
    # D = dif_central + dif_left + dif_right
    # D = [1:-1,:]
    #  version 2
    data = np.repeat([[1],[-2],[1]],n,axis=1)
    D = sparse.dia_matrix((data,[0,1,2]),shape=(n-2,n))
    return D




class Filter:
    def __init__(self,types='similarity',method='mean',type_matrix='D1',Nframe=0,winSize=150,overlap=30):
        self.winSize = winSize
        self.overlap= overlap
        self.stride = self.winSize - self.overlap
        self.Nframe = Nframe
        self.type = types
        self.method = method
        self.type_matrix = type_matrix
        self.alpha = alpha

        model_path = './deepfilter.pth'
        self.model = FilterNet()
        state = torch.load(model_path,map_location='gpu')
        self.model.load_state_dict(state['state_dict'])
        self.model.cuda()
        self.model.eval()


    def mean(self,inputs):
        halfSize = self.winSize/2
        smoothData = inputs.copy()
        for frame in range(self.Nframe-1):
            if frame < halfSize:
                start = int(0)
                end   = int(frame+halfSize+1)
                smoothData[frame] = np.mean(inputs[start : end])
            elif frame > self.Nframe-1 - halfSize:
                start = int(frame-halfSize )
                end   = int(self.Nframe-1)
                smoothData[frame] = np.mean(inputs[start : end])
            else:
                start = int(frame-halfSize )
                end   = int(frame+halfSize+1)
                smoothData[frame] = np.mean(inputs[start : end])
        return smoothData


    def HPfilter(self,inputs,lamda = 100):
        n = self.winSize+1
        halfSize = self.winSize/2
        smoothData = inputs.copy()
        if self.type_matrix =='D1':
            D = first_derivative_matrix(n)
        elif self.type_matrix =='D2':
            D = second_derivative_matrix(n)
        for frame in range(self.Nframe-1):
            if frame <= halfSize:
                start = int(0)
                end   = int(self.winSize+1)
                y = inputs[start : end]
                A = sparse.eye(n) + 2*lamda*D.T.dot(D)
                x = sparse.linalg.spsolve(A,y)
                smoothData[frame] = x[frame]
            elif frame >= self.Nframe-2 - halfSize:
                start = int(self.Nframe-2-self.winSize)
                end   = int(self.Nframe)
                y = inputs[start : end]
                A = sparse.eye(n) + 2*lamda*D.T.dot(D)
                x = sparse.linalg.spsolve(A,y)
                smoothData[self.Nframe-2 - int(halfSize):self.Nframe] = x[int(halfSize):self.winSize+1]
            else:
                start = int(frame-halfSize )
                end   = int(frame+halfSize+1)
                y = inputs[start : end]
                A = sparse.eye(n) + 2*lamda*D.T.dot(D)
                x = sparse.linalg.spsolve(A,y)
                smoothData[frame] = x[int(halfSize)]
        return smoothData


    def Mixfilter(self,inputs, lamda1, lamda2, alpha1, alpha2):
        idx=0
        maxIter = 100
        N = self.winSize
        m = N-1
        p = N-2
        rho = 10.0
        inputss = np.expand_dims(inputss, axis=1)
        D1 = first_deriv_matrix(N)
        D2 = second_deriv_matrix(N)
        z1 = np.zeros((m,1))
        z1_old = np.zeros((m,1))
        z2 = np.zeros((p,1))
        z2_old = np.zeros((p,1))
        u1 = np.zeros((m,1))
        u2 = np.zeros((p,1))
        #output = inputs.copy()
        I = np.identity(N,dtype=float)
        DTD1 = np.matmul(D1.T,D1)
        DTD2 = np.matmul(D2.T,D2)
        lamda1 = lamda1*alpha1
        lamda2 = lamda2*alpha2
        lamdaPrime1 = (1-alpha1)*lamda1
        lamdaPrime2 = (1-alpha2)*lamda2
        rho1 = rho*alpha1
        rho2 = rho*alpha2

        while(idx<maxIter):
            tmp1 = np.linalg.inv(I + rho1*DTD1 + rho2*DTD2 + 2*lamdaPrime1*DTD1 + 2*lamdaPrime2*DTD2)
            tmp2 = inputss + rho1*np.matmul(D1.T,(z1-u1)) + rho2*np.matmul(D2.T,(z2-u2))
            output =  np.matmul(tmp1,tmp2)

            #z update with relaxation
            z1_old = z1
            z2_old = z2
            A_hat1 = alpha1*D1.dot(inputss) + (1-alpha1)*z1_old
            tmp3 = A_hat1 + u1
            A_hat2 = alpha2*D2.dot(inputss) + (1-alpha2)*z2_old
            tmp4 = A_hat2 + u2
            z1 = soft_threshold(tmp3,lamda1/(rho1+0.0001))
            z2 = soft_threshold(tmp4,lamda2/(rho2+0.0001))
            #u-update
            u1 = u1 + A_hat1 - z1
            u2 = u2 + A_hat2 - z2
            #Termination checks
            # r1_norm = np.linalg.norm((D1.dot(inputs) - z1),2);
            # s1_norm = np.linalg.norm(ro1*D1.T.dot(z1-z1_old),2)
            # eps1_pri = math.sqrt(N)*ABSTOL + RELTOL*max(np.linalg.norm(D1.dot(inputs),2),np.linalg.norm(-z1,2));
            idx += 1
        return output

    def filter_L12(self,inputs,lamda1,lamda2):
        idx=0
        maxIter = 20
        N = self.winSize
        m = N-1
        p = N-2
        rho = 1.0
        alpha = 1
        ABSTOL = 1e-4
        RELTOL = 1e-2
        y = inputs.copy()
        z1 = np.zeros((m,1))
        z1_old = np.zeros((m,1))
        u1 = np.zeros((m,1))
        output = y.copy()
        I = np.identity(N,dtype=float)
        D1 = first_deriv_matrix(N)
        D2 = second_deriv_matrix(N)
        DTD1 = np.matmul(D1.T,D1)
        DTD2 = np.matmul(D2.T,D2)

        while(idx<maxIter):
            A1 = I + 2*lamda2*DTD2 + rho*DTD1
            tmp1 = np.linalg.inv(A1)
            tmp2 = y + rho*np.matmul(D1.T,(z1-u1))
            output = np.matmul(tmp1,tmp2)

            #z-update with relaxation
            z1_old = z1
            A_hat = alpha*D1.dot(y) + (1-alpha)*z1_old
            tmp3 = A_hat + u1
            z1 = soft_threshold(tmp3,lamda1/rho)

            #u-update
            u1 = u1 + A_hat - z1

            #Termination checks
            r_norm = np.linalg.norm(np.matmul(D1,output) - z1)
            s_norm = np.linalg.norm(-rho*np.matmul(D1.T,(z1-z1_old)))
            eps_pri =  math.sqrt(N)*ABSTOL + RELTOL*max(np.linalg.norm(np.matmul(D1,output)),np.linalg.norm(-z1));

            #print(r_norm,'  ',eps_pri)
            if r_norm<=eps_pri:
              break
            idx += 1
        return output


    def filter_L1(self,inputs,lamda):
        idx=0
        maxIter = 100
        N = self.winSize
        m = N-1
        p = N-2
        rho = 1
        alpha = 1
        ABSTOL = 1e-4
        RELTOL = 1e-2

        #y = np.expand_dims(inputs, axis=1)
        y = inputs.copy()
        #z = np.zeros((m,1))
        z = np.random.rand(m,1)
        u = np.zeros((m,1))
        x = np.zeros((N,1))

        I = np.identity(N,dtype=float)
        D1 = first_deriv_matrix(N)
        DTD1 = D1.T.dot(D1)


        while(idx<maxIter):
            a = (I + rho*DTD1)
            b = y + rho*np.matmul(D1.T,(z-u))
            x = np.linalg.solve(a, b)

            #z-update with relaxation
            z_old = z
            Ax_hat = alpha*D1.dot(x) + (1-alpha)*z_old
            z = soft_threshold(Ax_hat+u,lamda/rho)

            #u-update
            u = u + Ax_hat - z

            #Termination checks
            r_norm = np.linalg.norm(np.matmul(D1,x) - z)
            s_norm = np.linalg.norm(-rho*np.matmul(D1.T,(z-z_old)))
            eps_pri =  math.sqrt(N)*ABSTOL + RELTOL*max(np.linalg.norm(np.matmul(D1,x)),np.linalg.norm(-z));
            eps_dual = math.sqrt(N)*ABSTOL + RELTOL*np.linalg.norm(rho*np.matmul(D1.T,u))


            #print(r_norm,'  ',eps_pri)
            if r_norm<=eps_pri and s_norm<=eps_dual:
              break
            idx += 1
        return x    

    def deepFilter(self,startFrame,endFrame,inputs):


        trajectory  = torch.from_numpy(inputs.astype(float)).float().unsqueeze(1).cuda()
        smooth_path = smoothFilter(trajectory)
        result = smooth_path.cpu().data.numpy()
        return result



    def smooth_Trajectory(self,cameraPath):
        if self.type == 'similarity':
            trajectX,trajectY,trajectA,trajectS = cameraPath.reshape(-1,4).T
        else:
            trajectA,trajectB,trajectX,trajectC,trajectD,trajectY = cameraPath.reshape(-1,6).T

        smoothTraject = cameraPath.copy()

        Nstep = round(self.Nframe / self.stride)
        for idx in range(self.Nframe):
            startFrame = idx * self.stride
            endFrame = startFrame + self.winSize
            
            X = trajectX[startFrame:endFrame]
            Y = trajectY[startFrame:endFrame]
            A = trajectA[startFrame:endFrame]
            B = trajectB[startFrame:endFrame]
            C = trajectC[startFrame:endFrame]
            D = trajectD[startFrame:endFrame]
            
            X = torch.from_numpy(X.reshape(X.shape[0],1).astype(float)).float().cuda()
            Y = torch.from_numpy(Y.reshape(Y.shape[0],1).astype(float)).float().cuda()
            A = torch.from_numpy(A.reshape(A.shape[0],1).astype(float)).float().cuda()
            B = torch.from_numpy(B.reshape(B.shape[0],1).astype(float)).float().cuda()
            C = torch.from_numpy(C.reshape(C.shape[0],1).astype(float)).float().cuda()
            D = torch.from_numpy(D.reshape(D.shape[0],1).astype(float)).float().cuda()

            tmp_X = self.model(X)
            tmp_Y = self.model(Y)
            tmp_A = self.model(A)
            tmp_B = self.model(B)
            tmp_C = self.model(C)
            tmp_D = self.model(D)
            
            tmp_X = tmp_X.cpu().data.numpy()
            tmp_Y = tmp_Y.cpu().data.numpy()
            tmp_A = tmp_A.cpu().data.numpy()
            tmp_B = tmp_B.cpu().data.numpy()
            tmp_C = tmp_C.cpu().data.numpy()
            tmp_D = tmp_D.cpu().data.numpy()

            if idx == 0:    #first window
                smoothTraject[0:self.winSize-self.overlap,0] = tmp_A[0:self.winSize-self.overlap]
                smoothTraject[0:self.winSize-self.overlap,1] = tmp_B[0:self.winSize-self.overlap]
                smoothTraject[0:self.winSize-self.overlap,2] = tmp_X[0:self.winSize-self.overlap]
                smoothTraject[0:self.winSize-self.overlap,3] = tmp_C[0:self.winSize-self.overlap]
                smoothTraject[0:self.winSize-self.overlap,4] = tmp_D[0:self.winSize-self.overlap]
                smoothTraject[0:self.winSize-self.overlap,5] = tmp_Y[0:self.winSize-self.overlap]

                buffer_X = tmp_X[self.winSize-self.overlap:]
                buffer_Y = tmp_Y[self.winSize-self.overlap:]
                buffer_A = tmp_A[self.winSize-self.overlap:]
                buffer_B = tmp_B[self.winSize-self.overlap:]
                buffer_C = tmp_C[self.winSize-self.overlap:]
                buffer_D = tmp_D[self.winSize-self.overlap:]

            elif idx == Nstep-1:    #last window
                endFrame = self.Nframe-1
                startFrame = endFrame - self.winSize
                l = ((idx-1)*self.stride + self.winSize)-startFrame-1

                buffer_X = tmp_X[self.winSize-l:]
                buffer_Y = tmp_Y[self.winSize-l:]
                buffer_A = tmp_A[self.winSize-l:]
                buffer_B = tmp_B[self.winSize-l:]
                buffer_C = tmp_C[self.winSize-l:]
                buffer_D = tmp_D[self.winSize-l:]

                w1 = np.array([i/l for i in range(0,l)]).reshape(-1,1)
                w2 = np.array([1 - x for x in w1]).reshape(-1,1)

                tmpX_ = np.multiply(tmp_X[:l],w1) + np.multiply(buffer_X,w2)
                tmpY_ = np.multiply(tmp_Y[:l],w1) + np.multiply(buffer_Y,w2)
                tmpA_ = np.multiply(tmp_A[:l],w1) + np.multiply(buffer_A,w2)
                tmpB_ = np.multiply(tmp_B[:l],w1) + np.multiply(buffer_B,w2)
                tmpC_ = np.multiply(tmp_C[:l],w1) + np.multiply(buffer_C,w2)
                tmpD_ = np.multiply(tmp_D[:l],w1) + np.multiply(buffer_D,w2)

                smoothTraject[startFrame:startFrame+l,0] = tmpA_
                smoothTraject[startFrame:startFrame+l,1] = tmpB_
                smoothTraject[startFrame:startFrame+l,2] = tmpX_
                smoothTraject[startFrame:startFrame+l,3] = tmpC_
                smoothTraject[startFrame:startFrame+l,4] = tmpD_
                smoothTraject[startFrame:startFrame+l,5] = tmpY_

                smoothTraject[startFrame+l+1:,0] = tmp_A[l:]
                smoothTraject[startFrame+l+1:,1] = tmp_B[l:]
                smoothTraject[startFrame+l+1:,2] = tmp_X[l:]
                smoothTraject[startFrame+l+1:,3] = tmp_C[l:]
                smoothTraject[startFrame+l+1:,4] = tmp_D[l:]
                smoothTraject[startFrame+l+1:,5] = tmp_Y[l:]

            else:   #normal window
                w1 = np.array([i/self.overlap for i in range(0,self.overlap)]).reshape(-1,1)
                w2 = np.array([1 - j for j in w1]).reshape(-1,1)
                
                tmpX_ = np.multiply(tmp_X[:self.overlap],w1) + np.multiply(buffer_X,w2)
                tmpY_ = np.multiply(tmp_Y[:self.overlap],w1) + np.multiply(buffer_Y,w2)
                tmpA_ = np.multiply(tmp_A[:self.overlap],w1) + np.multiply(buffer_A,w2)
                tmpB_ = np.multiply(tmp_B[:self.overlap],w1) + np.multiply(buffer_B,w2)
                tmpC_ = np.multiply(tmp_C[:self.overlap],w1) + np.multiply(buffer_C,w2)
                tmpD_ = np.multiply(tmp_D[:self.overlap],w1) + np.multiply(buffer_D,w2)

                smoothTraject[startFrame:startFrame+self.overlap,0] = tmpA_
                smoothTraject[startFrame:startFrame+self.overlap,1] = tmpB_
                smoothTraject[startFrame:startFrame+self.overlap,2] = tmpX_
                smoothTraject[startFrame:startFrame+self.overlap,3] = tmpC_
                smoothTraject[startFrame:startFrame+self.overlap,4] = tmpD_
                smoothTraject[startFrame:startFrame+self.overlap,5] = tmpY_

                smoothTraject[startFrame+self.overlap:startFrame+self.winSize-self.overlap,0] = tmp_A[self.overlap:self.winSize-self.overlap]
                smoothTraject[startFrame+self.overlap:startFrame+self.winSize-self.overlap,1] = tmp_B[self.overlap:self.winSize-self.overlap]
                smoothTraject[startFrame+self.overlap:startFrame+self.winSize-self.overlap,2] = tmp_X[self.overlap:self.winSize-self.overlap]
                smoothTraject[startFrame+self.overlap:startFrame+self.winSize-self.overlap,3] = tmp_C[self.overlap:self.winSize-self.overlap]
                smoothTraject[startFrame+self.overlap:startFrame+self.winSize-self.overlap,4] = tmp_D[self.overlap:self.winSize-self.overlap]
                smoothTraject[startFrame+self.overlap:startFrame+self.winSize-self.overlap,5] = tmp_Y[self.overlap:self.winSize-self.overlap]
            
                buffer_X = tmp_X[self.winSize-self.overlap:]
                buffer_Y = tmp_Y[self.winSize-self.overlap:]
                buffer_A = tmp_A[self.winSize-self.overlap:]
                buffer_B = tmp_B[self.winSize-self.overlap:]
                buffer_C = tmp_C[self.winSize-self.overlap:]
                buffer_D = tmp_D[self.winSize-self.overlap:]
        
        return smoothPath





        # if self.type == 'similarity':
        #     vecX,vecY,vecA,vecS = cameraPath.reshape(-1,4).T
        #     data = (vecX,vecY,vecA,vecS)
        #     if self.method=='mean':
        #         smoothX,smoothY,smoothA,smoothS = map(self.mean, data)
        #     elif self.method=='L2':
        #         funcFilter = partial(self.HPfilter,type_matrix=self.type_matrix,alpha = self.alpha)
        #         smoothX,smoothY,smoothA,smoothS = map(funcFilter, data)
        #     smoothPath = np.stack([smoothX,smoothY,smoothA,smoothS],axis=1)
        #     return smoothPath

        # elif (self.type == 'affine') or (self.type == 'cnn_2'):
        #     a,b,x,c,d,y = cameraPath.reshape(-1,6).T
        #     data = (a,b,x,c,d,y)
        #     if self.method=='mean':
        #         aS,bS,xS,cS,dS,yS = map(self.mean, data)
        #     elif self.method=='L2':
        #         funcFilter = partial(self.HPfilter,type_matrix=self.type_matrix,alpha = self.alpha)
        #         aS,bS,xS,cS,dS,yS = map(funcFilter, data)
        #     smoothPath = np.stack([aS,bS,xS,cS,dS,yS],axis=1).reshape(-1,2,3)
        #     return smoothPath




















