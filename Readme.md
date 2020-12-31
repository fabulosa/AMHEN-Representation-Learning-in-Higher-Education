## Attributed Multiplex Heterogeneous Network (AMHEN) Representation Learning for Students and Courses in Higher Education

### [Publication](https://educationaldatamining.org/files/conferences/EDM2020/papers/paper_65.pdf) | [Presentation](https://www.youtube.com/watch?v=wy_oA1_zmWc&t=58s):
Evaluating sources of course information and models of representation on a variety of institutional prediction tasks, [Weijie Jiang](jennywjjiang.com) and [Zachary Pardos](https://gse.berkeley.edu/zachary-pardos). In *Proceedings of The 13th International Conference on Educational Data Mining (EDM 2020)*.

### Training on an enrollment dataset:
Preprocessed files required to train the AMHEN in our paper on an enrollment dataset:

* train.txt: Each line represents an edge (grade) connecting a student node and a course node, which contains three tokens <edge_type> <node1> <node2> where each token should be a number.

* node\_type.txt: Each line contains two tokens <node> <node_type>, where <node_type> should be consistent with the meta-path schema in the training command, i.e., --schema node\_type\_1-node\_type\_2-...-node\_type\_k-node\_type\_1. (Note that the first node type in the schema should equals to the last node type.) For enrollment data, only two types of nodes are supported, i.e., student node and course node. 

* feature.npy (optional, only feature of courses is supported currently): Each row represents features of a course, the number of which corresponds to course number.

### Prerequisites:
* python3
* pytorch
* install other dependencies by 
	* pip install -r requirements.txt

### Training Commands: 

* python3 src_attri/main.py
