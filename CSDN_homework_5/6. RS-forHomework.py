
# coding: utf-8

# In[1]:


# 将所有特征串联起来，构成RS_Train.csv
#RS_Test.csv
#为最后推荐系统做准备
from __future__ import division

import cPickle
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from numpy.random import random  
from collections import defaultdict

class RecommonderSystem:
  def __init__(self):
    # 读入数据做初始化
    
    #用户和活动新的索引
    self.userIndex = cPickle.load(open("PE_userIndex.pkl", 'rb'))
    self.eventIndex = cPickle.load(open("PE_eventIndex.pkl", 'rb'))
    self.n_users = len(self.userIndex)
    self.n_items = len(self.eventIndex)
    
    #用户-活动关系矩阵R
    #在train_SVD会重新从文件中读取,二者要求的格式不同，来不及统一了:(
    self.userEventScores = sio.mmread("PE_userEventScores").todense()
    
    #倒排表
    ##每个用户参加的事件
    self.itemsForUser = cPickle.load(open("PE_eventsForUser.pkl", 'rb'))
    ##事件参加的用户
    self.usersForItem = cPickle.load(open("PE_usersForEvent.pkl", 'rb'))
    
    #基于模型的协同过滤参数初始化,训练
    self.init_SVD()
    self.train_SVD(trainfile = "train.csv")
    
    #根据用户属性计算出的用户之间的相似度
    self.userSimMatrix = sio.mmread("US_userSimMatrix").todense()
    
    #根据活动属性计算出的活动之间的相似度
    self.eventPropSim = sio.mmread("EV_eventPropSim").todense()
    self.eventContSim = sio.mmread("EV_eventContSim").todense()
    
    #每个用户的朋友的数目
    self.numFriends = sio.mmread("UF_numFriends")
    #用户的每个朋友参加活动的分数对该用户的影响
    self.userFriends = sio.mmread("UF_userFriends").todense()
    
    #活动本身的热度
    self.eventPopularity = sio.mmread("EA_eventPopularity").todense()

  def init_SVD(self, K=20):
    #初始化模型参数（for 基于模型的协同过滤SVD_CF）
    self.K = K  
    
    #init parameters
    #bias
    self.bi = np.zeros(self.n_items)  
    self.bu = np.zeros(self.n_users)  
    
    #the small matrix
    self.P = random((self.n_users,self.K))/10*(np.sqrt(self.K))
    self.Q = random((self.K, self.n_items))/10*(np.sqrt(self.K))  
                  
          
  def train_SVD(self,trainfile = 'train.csv', steps=100,gamma=0.04,Lambda=0.15):
    #训练SVD模型（for 基于模型的协同过滤SVD_CF）
    #gamma：为学习率
    #Lambda：正则参数
    
    #偷懒了，为了和原来的代码的输入接口一样，直接从训练文件中去读取数据
    print "SVD Train..."
    ftrain = open(trainfile, 'r')
    ftrain.readline()
    self.mu = 0.0
    n_records = 0
    uids = []  #每条记录的用户索引
    i_ids = [] #每条记录的item索引
    #用户-Item关系矩阵R（内容同userEventScores相同），临时变量，训练完了R不再需要
    R = np.zeros((self.n_users, self.n_items))
    
    for line in ftrain:
        cols = line.strip().split(",")
        u = self.userIndex[cols[0]]  #用户
        i = self.eventIndex[cols[1]] #活动
        
        uids.append(u)
        i_ids.append(i)
        
        R[u,i] = int(cols[4])  #interested
        self.mu += R[u,i]
        n_records += 1
    
    ftrain.close()
    self.mu /= n_records
    
    # 请补充完整SVD模型训练过程
    for step in range(steps):
        RMSE_sum = 0.0
        rand_k = np.random.permutation(n_records)#打散数据样本（按行）
        for j in range(n_records):
            index = rand_k[j]
            u = uids[index]
            i = i_ids[index]
            e_ui = R[u,i] - self.pred_SVD(u,i)#残差
            RMSE_sum += e_ui**2               #残差平方和
            
            #随机梯度下降公式(对user和item的bias 是针对每个u和i的更新
            self.bu[u] += gamma*(e_ui-Lambda*self.bu[u])#公式 bu = bu - lambda*bu_gd
            self.bi[i] += gamma*(e_ui-Lambda*self.bi[i])#公式 bi = bi - lambda*bi_gd
            
            for k in range(self.K):
                self.P[u,k] += gamma*e_ui*self.Q[k,i] - Lambda*self.P[u,k]*gamma
                self.Q[k,i] += gamma*e_ui*self.P[u,k] - Lambda*self.Q[k,i]*gamma
        
        gamma = 1/np.sqrt(gamma)#学习率递减
    print "SVD trained"
    
  def pred_SVD(self, uid, i_id):
    #根据当前参数，预测用户uid对Item（i_id）的打分        
    ans=self.mu + self.bi[i_id] + self.bu[uid] + np.dot(self.P[uid,:],self.Q[:,i_id])  
        
    #将打分范围控制在0-1之间
    if ans>1:  
        return 1  
    elif ans<0:  
        return 0
    return ans  

  def sim_cal_UserCF(self, uid1, uid2 ):
    #请补充基于用户的协同过滤中的两个用户uid1和uid2之间的相似度（根据两个用户对item打分的相似度）
    si = dict() #为用户参加的相似的event的创建一个字典
    for item in self.itemsForUser[uid1]:
        if item in self.itemsForUser[uid2]:
            si[item] = 1
    num_cotem = len(si)#两个人共同参加event的个数
    if (si==0):         #检验相似度
        similarity = 0
        return similarity
    #用户uid1打过分的event
    s1 = []
    for item in si:
        s1.append(self.userEventScores[uid1,item])
    s1 = np.array(s1)    
    #用户uid2打过分的event
    s2 = []
    for item in si:
        s2.append(self.userEventScores[uid2,item])
    s2 = np.array(s2)
    
    #均值为0，所以不用减去打分均值 使用pearson 公式的另一个形式
    #(sumXY - sumXsumY/N)/sqrt(sumX2 - (sumX)2/N)sqrt(sumY2 - (sumY)2/N)
    s1_sum = np.sum(s1)
    s2_sum = np.sum(s2)
    s1_sum2=np.sum(s1**2)
    s2_sum2=np.sum(s2**2)
    
    s1s2_sum = np.sum(s1*s2)
    numerator = s1s2_sum-(s1_sum*s2_sum/num_cotem)
    denumerator = np.sqrt((s1_sum2 - s1_sum**2/num_cotem)*(s2_sum2 - s2_sum**2/num_cotem))
    if denumerator == 0:
        similarity = 0
        return 0

    
    similarity = numerator / denumerator
    
    
    
    return similarity  

  def userCFReco(self, userId, eventId):
    """
    根据User-based协同过滤，得到event的推荐度
    基本的伪代码思路如下：
    for item i
      for every other user v that has a preference for i
        compute similarity s between u and v
        incorporate v's preference for i weighted by s into running aversge
    return top items ranked by weighted average
    """
    #预测userId 对eventId感兴趣的程度
    #请补充完整代码
    u = self.userIndex[userId]
    i = self.eventIndex[eventId]
    
    sim_sum = 0.0
    rat_acc = 0.0
    #计算user 对 所有对该user关注的event的user的similarity
    for user in self.usersForItem[i]:
        sim = self.sim_cal_UserCF(user, u)
        if sim == 0: continue
        rat_acc += sim*self.userEventScores[user,i]
        sim_sum += sim
        
    if sim_sum == 0:
        return self.mu
    ans = rat_acc/sim_sum
     
    if ans > 1:
        return 1
    else:
        return 0
  
    return ans


  def sim_cal_ItemCF(self, i_id1, i_id2):
    #计算Item i_id1和i_id2之间的相似性
    #请补充完整代码
    si = dict() 
    for user in self.userForItem[i_id1]:
        if user in self.userForItem[i_id2]:
            si[user] = 1
    num_cotem = len(si)
    if (si==0):         #检验相似度
        similarity = 0
        return similarity
    #
    s1 = []
    for user in si:
        s1.append(self.userEventScores[user,i_id1])
    s1 = np.array(s1)    
    #
    s2 = []
    for user in si:
        s2.append(self.userEventScores[user,i_id2])
    s2 = np.array(s2)
    
    s1_sum = np.sum(s1)
    s2_sum = np.sum(s2)
    s1_sum2=np.sum(s1**2)
    s2_sum2=np.sum(s2**2)
    
    s1s2_sum = np.sum(s1*s2)
    numerator = s1s2_sum-(s1_sum*s2_sum/num_cotem)
    denumerator = np.sqrt((s1_sum2 - s1_sum**2/num_cotem)*(s2_sum2 - s2_sum**2/num_cotem))
    if denumerator == 0:
        similarity = 0
        return 0
    
    similarity = numerator / denumerator
    
    
    
    return similarity     
        
    return num/den  
            
  def eventCFReco(self, userId, eventId):    
    """
    根据基于物品的协同过滤，得到Event的推荐度
    基本的伪代码思路如下：
    for item i 
        for every item j tht u has a preference for
            compute similarity s between i and j
            add u's preference for j weighted by s to a running average
    return top items, ranked by weighted average
    """
    #请补充完整代码
    ans = 0.0
    u = self.userIndex[userId]
    i = self.eventIndex[eventId]
    
    sim_sum = 0.0
    rat_acc = 0.0
    for item in self.itemsForUser[u]:
        sim = self.sim_cal_UserCF(item, i)
        if sim == 0: continue
        rat_acc += sim*self.userEventScores[u,item]
        sim_sum += sim
        
    if sim_sum == 0:
        return self.mu
    ans = rat_acc/sim_sum
     
    if ans > 1:
        return 1
    else:
        return 0
   
    return ans
    
  def svdCFReco(self, userId, eventId):
    #基于模型的协同过滤, SVD++/LFM
    u = self.userIndex[userId]
    i = self.eventIndex[eventId]

    return self.pred_SVD(u,i)

  def userReco(self, userId, eventId):
    """
    类似基于User-based协同过滤，只是用户之间的相似度由用户本身的属性得到，计算event的推荐度
    基本的伪代码思路如下：
    for item i
      for every other user v that has a preference for i
        compute similarity s between u and v
        incorporate v's preference for i weighted by s into running aversge
    return top items ranked by weighted average
    """
    i = self.userIndex[userId]
    j = self.eventIndex[eventId]

    vs = self.userEventScores[:, j]
    sims = self.userSimMatrix[i, :]

    prod = sims * vs

    try:
      return prod[0, 0] - self.userEventScores[i, j]
    except IndexError:
      return 0

  def eventReco(self, userId, eventId):
    """
    类似基于Item-based协同过滤，只是item之间的相似度由item本身的属性得到，计算Event的推荐度
    基本的伪代码思路如下：
    for item i 
      for every item j that u has a preference for
        compute similarity s between i and j
        add u's preference for j weighted by s to a running average
    return top items, ranked by weighted average
    """
    i = self.userIndex[userId]
    j = self.eventIndex[eventId]
    js = self.userEventScores[i, :]
    psim = self.eventPropSim[:, j]
    csim = self.eventContSim[:, j]
    pprod = js * psim
    cprod = js * csim
    
    pscore = 0
    cscore = 0
    try:
      pscore = pprod[0, 0] - self.userEventScores[i, j]
    except IndexError:
      pass
    try:
      cscore = cprod[0, 0] - self.userEventScores[i, j]
    except IndexError:
      pass
    return pscore, cscore

  def userPop(self, userId):
    """
    基于用户的朋友个数来推断用户的社交程度
    主要的考量是如果用户的朋友非常多，可能会更倾向于参加各种社交活动
    """
    if self.userIndex.has_key(userId):
      i = self.userIndex[userId]
      try:
        return self.numFriends[0, i]
      except IndexError:
        return 0
    else:
      return 0

  def friendInfluence(self, userId):
    """
    朋友对用户的影响
    主要考虑用户所有的朋友中，有多少是非常喜欢参加各种社交活动/event的
    用户的朋友圈如果都积极参与各种event，可能会对当前用户有一定的影响
    """
    nusers = np.shape(self.userFriends)[1]
    i = self.userIndex[userId]
    return (self.userFriends[i, :].sum(axis=0) / nusers)[0,0]

  def eventPop(self, eventId):
    """
    本活动本身的热度
    主要是通过参与的人数来界定的
    """
    i = self.eventIndex[eventId]
    return self.eventPopularity[i, 0]



# In[2]:


def generateRSData(RS, train=True, header=True):
    """
    把前面user-based协同过滤 和 item-based协同过滤，以及各种热度和影响度作为特征组合在一起
    生成新的训练数据，用于分类器分类使用
    """
    fn = "train.csv" if train else "test.csv"
    fin = open(fn, 'rb')
    fout = open("RS_" + fn, 'wb')
    
    #忽略第一行（列名字）
    fin.readline().strip().split(",")
    
    # write output header
    if header:
      ocolnames = ["invited", "userCF_reco", "evtCF_reco","svdCF_reco","user_reco", "evt_p_reco",
        "evt_c_reco", "user_pop", "frnd_infl", "evt_pop"]
      if train:
        ocolnames.append("interested")
        ocolnames.append("not_interested")
      fout.write(",".join(ocolnames) + "\n")
    
    ln = 0
    for line in fin:
      ln += 1
      if ln%500 == 0:
          print "%s:%d (userId, eventId)=(%s, %s)" % (fn, ln, userId, eventId)
          #break;
      
      cols = line.strip().split(",")
      userId = cols[0]
      eventId = cols[1]
      invited = cols[2]
      
      userCF_reco = RS.userCFReco(userId, eventId)
      itemCF_reco = RS.eventCFReco(userId, eventId)
      svdCF_reco = RS.svdCFReco(userId, eventId)
        
      user_reco = RS.userReco(userId, eventId)
      evt_p_reco, evt_c_reco = RS.eventReco(userId, eventId)
      user_pop = RS.userPop(userId)
     
      frnd_infl = RS.friendInfluence(userId)
      evt_pop = RS.eventPop(eventId)
      ocols = [invited, userCF_reco, itemCF_reco, svdCF_reco,user_reco, evt_p_reco,
        evt_c_reco, user_pop, frnd_infl, evt_pop]
      
      if train:
        ocols.append(cols[4]) # interested
        ocols.append(cols[5]) # not_interested
      fout.write(",".join(map(lambda x: str(x), ocols)) + "\n")
    
    fin.close()
    fout.close()


# In[3]:


RS = RecommonderSystem()
print "生成训练数据...\n"
generateRSData(RS,train=True,  header=True)

print "生成预测数据...\n"
generateRSData(RS, train=False, header=True)


# 时间、地点等特征都没有处理了，可以考虑用户看到event的时间与event开始时间的差、用户地点和event地点的差异。。。
