import DataPreprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import GRUCell
from tqdm import tqdm
from attention import attention_mechanism
from gensim.models import Word2Vec

class LR(DataPreprocess.dataprocess):
    def __init__(self, batch_size=256, vector_size=100, max_seq=25, learn_rate=0.05, output_p=0.5, num_layers = 2, num_hidden = 64):
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.max_seq = max_seq
        self.learn_rate = learn_rate
        self.output_p = output_p
        self.num_layers = num_layers
        self.num_hidden = num_hidden

    # Method to fetch the data
    def fetchdata(self, file, filepath='', dump_flag = False, combine_flag = False, w2vflg = False):
        if dump_flag is True:
            self.filedump(file, filepath)

        data = self.loadjson(file)

        # Check if the responses and parent have to combined or not
        if combine_flag is True:
            sentences, labels = self.combine(data)
            labels = list(map(int, labels))
            labels = list(map(self.score_setup, labels))
            return sentences, np.array(labels)
        else:
            context = []
            idx = 0
            for elem in data['parent']:
                for _ in range(len(data['response'][idx])):
                    context.append(elem[-1])
                idx = idx + 1
            response = [x for y in data['response'] for x in y]
            labels = [x for y in data['labels'] for x in y]


        # ##### Testing purpose ####
        # response = response[0:128]
        # context = context[0:128]
        # labels = labels[0:128]
        
        # labels = list(map(int, labels))
        # labels = np.array(list(map(self.score_setup, labels)))

        return context, response, labels


    def convert(self, context, glove_map):
        # Convert the sentences into a sequence

        context = np.stack([self.fit_to_size(self.sentence2sequence(sent, glove_map, 2* self.max_seq,combine_flag = True),
            self.max_seq * self.vector_size * 2,combine_flag = True) for sent in context])

        return context


    # Method to fit the lr Model
    def fit(self, context, labels, context_test = None, labels_test = None,test_flg = False, w2vflg=True):
        # Create Placeholders
        x_data = tf.placeholder(tf.float32, [None, 2*self.max_seq*self.vector_size])
        y_target = tf.placeholder(tf.float32, [None, self.num_layers]) 

        # Set model weights
        W = tf.Variable(tf.zeros([2*self.max_seq*self.vector_size, 2]))
        b = tf.Variable(tf.zeros([2]))

        # Construct model
        prediction = tf.nn.sigmoid(tf.matmul(x_data, W) + b) # Softmax
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Minimize error using cross entropy
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
                    logits = prediction, labels = y_target))
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(self.learn_rate).minimize(loss)
    

        # Function to save the current trained model
        saver = tf.train.Saver()
        sess = tf.Session()
        training_iterations = int(len(context) / self.batch_size)
        # training_iterations = range(0, len(response), self.batch_size)
        # training_iterations = tqdm(training_iterations)
        test_accuracy = []
        test_confuse = [[0,0],[0,0]]

        ## For Plotting
        test_accuracy = []
        accuracy_plot = []
        train_error_plot = []
        loss_error_plot = []
        train_plot = []
        loss_plot = []
        x_test = []
        x_train = []

        train_plot_epoch = []
        loss_plot_epoch = []
        x_test = []
        x_train_epoch = []


        if w2vflg is False:
            # Fetch the Glove vectors
            glove_map = self.fetchglove()
        else:
            # Fetch the word2vec
            glove_map = Word2Vec.load('w2vmodel')

        if test_flg is False:
            # Training
            # file = open("result_LR_UB.txt","w") 
            # file.write(str(training_iterations) + "\n")
            print(training_iterations)
            init = tf.global_variables_initializer()
            sess.run(init)
            batch = 0
            for i in range(training_iterations):
                #batch = np.random.randint(len(response), size=self.batch_size)
                # if i != 0:
                #     prev = batch
                #     batch = prev + self.batch_size
                # else:
                #     prev = 0
                #     batch change value in tf array= self.batch_size

                cont = self.convert(context[i*self.batch_size:(i+1)*self.batch_size], glove_map)
                cont_p, label_p = (cont,labels[i*self.batch_size:(i+1)*self.batch_size])

                sess.run([optimizer], feed_dict={x_data: cont_p, y_target: label_p})
                if i % 10 == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x_data: cont_p, y_target: label_p})
                    # Calculate batch loss
                    tmp_loss = sess.run(loss, feed_dict={x_data: cont_p, y_target: label_p})
                    # Display results
                    # print(tmp_loss)
                    # print(acc)
                    print("Iter " + str(i) + ", Minibatch Loss= " + \
                          "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc))
                    # file.write("Iter " + str(i) + ", Minibatch Loss= " + \
                    #       "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                    #       "{:.5f}".format(acc) + "\n" )

                    train_plot.append(acc)
                    loss_plot.append(tmp_loss)
                    x_train.append(i)
                    # if(i < test_iterations):
                    cont = self.convert(context_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size], glove_map)
                    cont_p, label_p = (cont, labels_test[(i%250)*self.batch_size:(i%250+1)*self.batch_size])

                    # Calculate batch accuracy
                    acc_test = sess.run(accuracy, feed_dict={x_data: cont_p,  y_target: label_p})
                    # Calculate batch loss
                    loss_test = sess.run(loss, feed_dict={x_data: cont_p, y_target: label_p})
                    print("Iter " + str(i) + ", Minibatch Loss Test= " + \
                    "{:.6f}".format(loss_test) + ", Training Accuracy Test= " + \
                    "{:.5f}".format(acc_test))

                    train_error_plot.append([acc,acc_test])
                    loss_error_plot.append([tmp_loss,loss_test])
                    x_test.append(i)


            np.save("Graph/train_error_plot_LR_1",train_error_plot)
            np.save("Graph/loss_error_plot_LR_1",loss_error_plot)
            np.save("Graph/x_test_LR_1",x_test)


            saver.save(sess, 'ModelEpoch/Final_model_LR_1')
            print('Model saved to Root Directory of the python file')
        else:
            confusion_matrix_tf = tf.confusion_matrix(tf.cast(tf.argmax(y_target, 1), 'int32'), tf.cast(tf.argmax(prediction, 1), 'int32'))
            saver.restore(sess, 'ModelEpoch/Final_model_LR_1')
            batch = 0
            print(training_iterations)

            for i in range(training_iterations):
            
                cont = self.convert(context[i*self.batch_size:(i+1)*self.batch_size], glove_map)
                cont_p, label_p = (cont, labels[i*self.batch_size:(i+1)*self.batch_size])
                predict = sess.run(accuracy, feed_dict={x_data: cont_p, y_target: label_p})
                cm = sess.run(confusion_matrix_tf, feed_dict={x_data: cont_p, y_target: label_p})
                if i % 10 == 0:
                    print("Iter " + str(i) +  ", Training Accuracy= " + \
                          "{:.5f}".format(predict))
                    # print(cm)
                test_confuse = np.add(test_confuse, cm)
                test_accuracy.append(predict)
                    # print(predict)

            print("Accuracy: ",float(sum(test_accuracy)/len(test_accuracy)))
            precision = test_confuse[1][1] / (test_confuse[0][1] + test_confuse[1][1])
            recall = test_confuse[1][1] / (test_confuse[1][0] + test_confuse[1][1])
            print("Confuse: ")
            print(test_confuse)
            print("Precision: ",precision)
            print("Recall: ",recall)
            print("F1-Score: ",2*(precision * recall) / (precision + recall))

        sess.close()


lr = LR()
train = input("Would you like to train the data set? (Y/N)")
if train == 'Y':
    context, labels = lr.fetchdata('train_data.json', combine_flag = True, w2vflg=True)
    context_test, labels_test = lr.fetchdata('test_data.json', w2vflg=True, combine_flag = True)
    lr.fit(context, labels, context_test, labels_test ,w2vflg=True,  test_flg = False)
elif train == 'N':
    context, labels = lr.fetchdata('test_data.json', w2vflg=True, combine_flag = True)
    lr.fit(context, labels, test_flg=True, w2vflg=True)
    #lr.test(context, response, labels, prediction)
