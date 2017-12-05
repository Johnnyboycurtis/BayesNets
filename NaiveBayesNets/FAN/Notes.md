# Forest Augmented Naive Bayes

## ABSTRACT
Naive Bayes has been widely used in data mining as a simple and effective classification algorithm. Since its conditional independence assumption is rarely true, numerous algorithms have been proposed to improve naive Bayes, among which tree augmented naive Bayes (TAN) [3] achieves a significant improvement in term of classification accuracy, while maintaining efficiency and model simplicity. In many real-world data mining applications, however, an accurate ranking is more desirable than a classification. Thus it is interesting whether TAN also achieves significant improvement in term of ranking, measured by AUC(the area under the Receiver Operating Characteristics curve) [8,1]. Unfortunately, our experiments show that TAN performs even worse than naive Bayes in ranking. Responding to this fact, we present a novel learning algorithm, called forest augmented naive Bayes (FAN), by modifying the traditional TAN learning algorithm. We experimentally test our algorithm on all the 36 data sets recommended by Weka [12], and compare it to naive Bayes, SBC [6], TAN [3], and C4.4 [10], in terms of AUC. The experimental results show that our algorithm outperforms all the other algorithms significantly in yielding accurate rankings. Our work provides an effective and efficient data mining algorithm for applications in which an accurate ranking is required.





