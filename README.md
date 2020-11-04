Text similarity
---------
This module aims to provide an easy way to test diferent text similarity techniques. 
Altought there is more powerfull techniques nowadays,  approximations based on wordNet 
and word2vect are still used.

For evaluating the different methods we used the STS (Semantic Textual Similarity) dataset 
used in the SemEval task. We measure the effectiveness of each method with the correlation
of the given similarity and the one calculated by our module

As the general rule word2vect has shown better results that wordNet approaches.
