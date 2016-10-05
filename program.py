#!/usr/bin/env python2.7
#encoding:utf-8
import os,sys
import numpy as np
#pip install editdistance
#https://pypi.python.org/pypi/editdistance
#amoco construct cfg, install from source code
#https://github.com/bdcht/amoco
import amoco,editdistance
from amoco.main import *
import hashlib
# Use sqlalchemy and sqlite to store malware signatures
#http://www.sqlalchemy.org
#http://docs.sqlalchemy.org/en/rel_1_0/orm/tutorial.html
from sqlalchemy import create_engine, MetaData
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy.sql import select,func

class BkTree():
    "A Bk-Tree implementation."
     
    def __init__(self, root):
        self.root = root 
        self.tree = (root, {})
     
    def build(self, words):
        "Build the tree."
        for word in words:
            self.tree = self.insert(self.tree, word)

    def insert(self, node, word):
        "Inserts a word in the tree."
        d = editdistance.eval(word, node[0])
#        print node[1]
        if d not in node[1]:
            node[1][d] = (word, {})
        else:
            self.insert(node[1][d], word)
        return node

    def query(self, word, n):
        "Returns a list of words that have the specified edit distance from the search word."
        def search(node):
            d = editdistance.eval(word, node[0])
            results = []
            if d == n:
                results.append(node[0])
            for i in range(d-n, d+n+1):
                children = node[1]
                if i in children:
                    results.extend(search(node[1][i]))
            return results
         
        root = self.tree
        return search(root)

class DBOperator():
    def __init__(self):
	self.engine = create_engine('sqlite:///malwareVNF.db')
	self.metadata = MetaData()
        self.metadata.bind = self.engine
        self.metadata.reflect()
        self.conn = self.engine.connect()

    def reflectTB(self, tb_name):
        return Table(tb_name, self.metadata, autoload=True, autoload_with=self.engine)
   
    def drop(self, table_t):
        i = table_t.drop()
        r = self.conn.execute(i)
        return r
     
    def insert(self, table_t, dict_u):
    	i = table_t.insert()
        r = self.conn.execute(i, **dict_u)
        return r

    def select(self, table_t):
        s = select([table_t])
        r = self.conn.execute(s)
        return r.fetchall()
    
    def exactSelect(self, table_t, str_sig):
        s = select([table_t]).where(table_t.c.signature == str_sig)
        r = self.conn.execute(s)
        return r

    def count(self, table_t):
        s = select([func.count(table_t.c.id)])
        r = self.conn.execute(s)
        return r.fetchall()[0][0]        

class MalwareDB():
    def __init__(self):
        self.engine = create_engine('sqlite:///malwareVNF.db')
        self.metadata = MetaData()
        self.metadata.bind = self.engine
        self.metadata.reflect()
        self.conn = self.engine.connect()
        self.db_op = DBOperator()

    def createGlobalTB(self): 
	if 'global_t' not in self.metadata.tables:
            self.global_t = Table('global_t', self.metadata,
                              Column('id', Integer, primary_key=True),
                              Column('digest', String),
                              Column('signature', String)) 
            self.metadata.create_all(self.engine)
        else:
            self.global_t = Table('global_t', self.metadata, autoload=True, autoload_with=self.engine)    

    def createLocalTB(self, tb_name):
        if tb_name not in self.metadata.tables: 
    	    local_t = Table(tb_name, self.metadata,
                            Column('id', Integer, primary_key=True),
                            Column('sig_fun', String))
    	    self.metadata.create_all(self.engine)
        else:
            #print tb_name + '.db is already created!'
            local_t = Table(tb_name, self.metadata, autoload=True, autoload_with=self.engine)
        return local_t

    def storeLocalSignatures(self, dir_prog):
        if dir_prog not in self.metadata.tables:
            local_t = self.createLocalTB(dir_prog)
            nfv_sig = NFVSignature(dir_prog)
            str_sig, vnf_functions = nfv_sig.getcfg()
            for func in vnf_functions:
                u = dict(sig_fun = cfg.signature(func.cfg))
                self.db_op.insert(local_t, u)
        else:
            print dir_prog + ' is already in the malware Database!'
            local_t = Table(dir_prog, self.metadata, autoload=True, autoload_with=self.engine)
        return local_t
     
    def storeSignatures(self, dir_prog):
        self.createGlobalTB()

        nfv_sig = NFVSignature(dir_prog)
        str_sig, vnf_functions = nfv_sig.getcfg()
        
        str_hash = hashlib.sha512(str_sig).hexdigest()
        hash_set = self.db_op.select(self.global_t.c.digest)
        if (str_hash,) not in hash_set:
            u = dict(digest = str_hash, signature = str_sig)
            self.db_op.insert(self.global_t,u)
        
            local_t = self.createLocalTB(dir_prog)
            for func in vnf_functions:
                u = dict(sig_fun = cfg.signature(func.cfg))
                self.db_op.insert(local_t, u)
        else:
            print dir_prog + ' already in the database!'

    def buildVNFDB(self):
        f_mal = open('malware.txt','r')
        dir_vnf = '../amoco/tests/samples/'

        for line in f_mal.readlines():
            self.storeSignatures(dir_vnf + line.strip())
 
class NFVSignature():
    def __init__(self, dir_prog):
        self.prog = amoco.system.loader.load_program(dir_prog)
        
    def getcfg(self):
        z = amoco.lsweep(self.prog)
        z.getcfg()
        str_sig = ''
        for func in z.functions():
            #print cfg.signature(fun.cfg)
            str_sig = str_sig + cfg.signature(func.cfg)
        #print hashlib.sha512(str_sig).hexdigest()     
        return str_sig, z.functions() 

class ApproximateMatch():
    def __init__(self):
        self.threshold_func = 0.9
        self.threshold_prog = 0.6
        self.db_op = DBOperator()
        self.mal_db = MalwareDB()

    def calcuWeight(self, str_func, str_sig):
        return len(str_func)/(len(str_sig)+0.0)
    
    def calcuSimilarity(self, str_x, str_y):
        ed = editdistance.eval(str_x, str_y)
        print 'editdistance (' + str_x + ',' + str_y +'):' + str(ed)
        return 1 - (ed/(max(len(str_x),len(str_y))+0.0))
    
    def calcuAsymmetricSim(self, table_x, table_y):
        str_funcs = []
        for i,str_func in self.db_op.select(table_y):
            str_funcs.append(str_func)
       # print str_funcs
        bk_tree = BkTree(str_funcs[0])
        del str_funcs[0]
        bk_tree.build(str_funcs)

        x_len = 0
        for i,str_func in self.db_op.select(table_x):
            x_len = x_len + len(str_func)

        Similar_set = []
        Asym_similarity = 0
        for i,str_func in self.db_op.select(table_x):
            W_weight = len(str_func)/(x_len+0.0)
            E_error  = int(len(str_func)*(1-self.threshold_func)) + 1
#            print 'W_weight:' + str(W_weight)
            print 'E_error:' + str(E_error)
            for str_node in bk_tree.query(str_func, E_error):                     
                if str_node not in Similar_set:
                    similarity_rate = self.calcuSimilarity(str_func, str_node) 
                    print 'func similarity_rate:' + str(similarity_rate) 
                    if similarity_rate >= self.threshold_func:
                        Similar_set.append(str_node)
                        Asym_similarity = Asym_similarity + W_weight * similarity_rate
                        break
        return Asym_similarity
    
    def calcuProgramSim(self, dir_prog):
        Program_sims = []
        max_sim = 0
        f_mal = open('malware.txt','r')
        dir_vnf = '../amoco/tests/samples/'

        test_tb = self.mal_db.storeLocalSignatures(dir_prog)
#        print test_tb
        print self.db_op.select(test_tb)
        for line in f_mal.readlines():
            print "Comparing with " + line.strip()
            local_tb = self.db_op.reflectTB(dir_vnf + line.strip())
            print self.db_op.select(local_tb)
            s_x = self.calcuAsymmetricSim(test_tb, local_tb)
            print 'Asym(test, local): ' + str(s_x)
            s_y = self.calcuAsymmetricSim(local_tb, test_tb)
            print 'Asym(local,test): ' + str(s_y)
            temp_sim = s_x * s_y
            print 'Similarity: ' + str(temp_sim)
            Program_sims.append(temp_sim)       
            if max_sim < temp_sim:
               max_sim = temp_sim

#        self.db_op.drop(test_tb)
 
        print 'Similarity Ratio between ' + dir_prog + ' and VNFs in the MalwareDB: '
        print max_sim

        if max_sim >= self.threshold_prog:
           print "The tested VNF is malicious!"
        else:
           print "The tested VNF is secure at present!"


  
if __name__ == "__main__":
    mal_db = MalwareDB()
    mal_db.buildVNFDB()

    am =  ApproximateMatch()
    
    am.calcuProgramSim('../amoco/tests/samples/x86/cpflow.elf')


     
