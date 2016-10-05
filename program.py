#!/usr/bin/env python2.7
#encoding:utf-8
import os,sys
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
        print node[1]
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
	self.engine = create_engine('sqlite:///foo.db')
	self.metadata = MetaData()
        self.metadata.bind = self.engine
        self.metadata.reflect()
        self.conn = self.engine.connect()

#    def createLocalTB(self, l_name):
#    	local_t = Table(l_name, self.metadata,
#                        Column('id', Integer, primary_key=True),
#                        Column('sig_fun', String))
#
#    	self.metadata.create_all(self.engine)
#    	self.conn = self.engine.connect()
#    	return local_t
    def reflectTB(self, tb_name):
        return Table(tb_name, self.metadata, autoload=True, autoload_with=self.engine)
    
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
        self.engine = create_engine('sqlite:///foo.db')
        self.metadata = MetaData()
        self.metadata.bind = self.engine
        self.metadata.reflect()
        self.conn = self.engine.connect()
        self.db_op = DBOperator()

    def createGlobalTB(self): 
	if 'global_t' not in self.metadata.tables:
            global_t = Table('global_t', self.metadata,
                              Column('id', Integer, primary_key=True),
                              Column('digest', String),
                              Column('signature', String)) 
            self.metadata.create_all(self.engine)
        else:
            global_t = Table('global_t', self.metadata, autoload=True, autoload_with=self.engine)
        
        u = dict(digest='di', signature='123')
        self.db_op.insert(global_t, u)

#    def reflectGlobalTB(self):
#        global_t = Table('global_t', self.metadata, autoload=True, autoload_with=self.engine)
#        print global_t.columns
#        u = dict(digest='di', signature='123')
#        self.db_op.insert(global_t, u)
#        
#        print self.conn.execute(select([global_t])).fetchall()
  
    def createLocalTB(self, tb_name):
        if tb_name not in self.metadata.tables: 
    	    local_t = Table(tb_name, self.metadata,
                            Column('id', Integer, primary_key=True),
                            Column('sig_fun', String))
    	    self.metadata.create_all(self.engine)
        else:
            local_t = Table(tb_name, self.metadata, autoload=True, autoload_with=self.engine)
        

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
        print hashlib.sha512(str_sig).hexdigest()     
        return str_sig, z.functions() 

class ApproximateMatch():
    def __init__(self, db_op):
        self.threshold_func = 0.9
        self.threshold_prog = 0.6
        self.db_op = db_op

    def calcuWeight(self, str_func, str_sig):
        return len(str_func)/(len(str_sig)+0.0)
    
    def calcuSimilarity(self, str_x, str_y):
        return 1 - (editdistance.eval(str_x, str_y)/(max(len(str_x),len(str_y))+0.0))
    
    def calcuAsymmetricSim(self, table_x, table_y):
        bk_tree = BkTree(root_y)
        str_funcs = []
        for i,str_func in self.db_op.select(table_y):
            str_funcs.append(str_func)
        bk_tree.build(str_funcs)

        x_len = 0
        for i,str_func in self.db_op.select(table_x):
            x_len = x_len + len(str_func)

        Similar_set = []
        Asym_similarity = 0
        for i,str_func in self.db_op.select(table_x):
            W_weight = len(str_func)/(x_len+0.0)
            E_error  = int(len(str_func)*(1-self.threshold_func))

            for str_node in bk_tree.query(str_func, E_error):                     
                if str_node not in Similar_set:
                    similarity_rate = self.calcuSimilarity(str_func, str_node)  
                    if similarity_rate >= self.threshold:
                        Similar_set.append(str_node)
                        Asym_similarity = Asym_similarity + W_weight * similarity_rate
                        break
        return Asym_similarity
    
    def calcuProgramSim(self, dir_prog):
        test_tb = self.db_op.createLocalTB(dir_prog)
        Program_sims = []
        max_sim = 0
        for i in range(0, self.db_op.count(db_op.global_t)):
            s_x = self.calcuAsymmetricSim(test_tb, local_tb[i])
            s_y = self.calcuAsymmetricSim(local_tb[i], test_tb)
            temp_sim = s_x * s_y
            Program_sims.append(temp_sim)       
            if max_sim < temp_sim:
               max_sim = temp_sim

        if max_sim >= self.threshold_prog:
           print "The tested VNF is malicious!"
  
if __name__ == "__main__":
    DBOperator()
    mal_db = MalwareDB()
    mal_db.createGlobalTB()
    mal_db.reflectGlobalTB()

#    db_op = DBOperator()
#    lt1 = db_op.createLocalTB('local_t')
#    u = dict(digest='di',signature='123')
#    u1 = dict(sig_fun='12345')
#    db_op.insert(lt1, u1)
#    db_op.insert(db_op.global_t, u)
#    r = db_op.select(lt1)
#    #X =[]
#    #for i,j in r:
#    #   X.append(j)
#    #print X
#    r = db_op.exactSelect(db_op.global_t, '12')
#    print r
#    db_op.count(lt1)
#
#    sig_ob = NFVSignature('samples/x86/flow.elf')
#    print sig_ob.getcfg()
#    print max(1,2)
#
#    root='cat'
#    bk_tree=BkTree(root)
#    words=['bat','cat','bats']
#    bk_tree.build(words)
#    print bk_tree.query('car',1)
     
