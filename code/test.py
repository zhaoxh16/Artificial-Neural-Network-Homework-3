import tensorflow as tf

# 创建writer对象
writer = tf.summary.FileWriter("/path/to/metadata_logs", tf.get_default_graph())

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
        if i % 1000 == 0:
            # 这里通过trace_level参数配置运行时需要记录的信息，
            # tf.RunOptions.FULL_TRACE代表所有的信息
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # 运行时记录运行信息的proto，pb是用来序列化数据的
            run_metadata = tf.RunMetadata()
            # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销信息
            _, loss_value, step = sess.run(
                [train_op, loss, global_step],
                feed_dict={x: x_batch, y_: y_batch},
                options=run_options, run_metadata=run_metadata)
            # 将节点在运行时的信息写入日志文件
            writer.add_run_metadata(run_metadata, 'step %03d' % i)
        else:
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

writer.close()
