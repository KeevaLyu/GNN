import os
from collections import namedtuple, defaultdict
from collections.abc import Iterable
from timeit import default_timer

from dataclasses import dataclass
from numpy import zeros, float32 as REAL, vstack, integer, dtype
import numpy as np

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.utils import deprecated
from gensim.models import Word2Vec, FAST_VERSION  # noqa: F401
from gensim.models.keyedvectors import KeyedVectors, pseudorandom_weak_vector

 class Doc2Vec(Word2Vec):
     def __init__(
             self, documents=None, corpus_file=None, vector_size=100, dm_mean=None, dm=1, dbow_words=0, dm_concat=0,
            dm_tag_count=1, dv=None, dv_mapfile=None, comment=None, trim_rule=None, callbacks=(),
            window=5, epochs=10, shrink_windows=True, **kwargs,
     ):
        """Class for training, using and evaluating neural networks described in
        `Distributed Representations of Sentences and Documents <http://arxiv.org/abs/1405.4053v2>`_.

        Parameters
        ----------
        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            Input corpus, can be simply a list of elements, but for larger corpora,consider an iterable that streams
            the documents directly from disk/network. If you don't supply `documents` (or `corpus_file`), the model is
            left uninitialized -- use if you plan to initialize it in some other way.
        corpus_file : str, optional
            Path to a corpus file in :class:`~gensim.models.word2vec.LineSentence` format.
            You may use this argument instead of `documents` to get performance boost. Only one of `documents` or
            `corpus_file` arguments need to be passed (or none of them, in that case, the model is left uninitialized).
            Documents' tags are assigned automatically and are equal to line number, as in
            :class:`~gensim.models.doc2vec.TaggedLineDocument`.
        dm : {1,0}, optional
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        vector_size : int, optional
            Dimensionality of the feature vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling.
            In Python 3, reproducibility between interpreter launches also requires use of the `PYTHONHASHSEED`
            environment variable to control hash randomization.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        epochs : int, optional
            Number of iterations (epochs) over the corpus. Defaults to 10 for Doc2Vec.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        dm_mean : {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean.
            Only applies when `dm` is used in non-concatenative mode.
        dm_concat : {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input
            is no longer the size of one (sampled or arithmetically combined) word vector, but the
            size of the tag(s) and all words in the context strung together.
        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using
            dm_concat mode.
        dbow_words : {1,0}, optional
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training; If 0, only trains doc-vectors (faster).
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.
        shrink_windows : bool, optional
            New in 4.1. Experimental.
            If True, the effective window size is uniformly sampled from  [1, `window`]
            for each target word during training, to match the original word2vec algorithm's
            approximate weighting of context words by distance. Otherwise, the effective
            window size is always fixed to `window` words to either side.

        Some important internal attributes are the following:

        Attributes
        ----------
        wv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object essentially contains the mapping between words and embeddings. After training, it can be used
            directly to query those embeddings in various ways. See the module level docstring for examples.

        dv : :class:`~gensim.models.keyedvectors.KeyedVectors`
            This object contains the paragraph vectors learned from the training data. There will be one such vector
            for each unique document tag supplied during training. They may be individually accessed using the tag
            as an indexed-access key. For example, if one of the training documents used a tag of 'doc003':
            .. sourcecode:: pycon
            >>> model.dv['doc003']
        """

        corpus_iterable = documents

        if dm_mean is not None:
            self.cbow_mean = dm_mean

        self.dbow_words = int(dbow_words)
        self.dm_concat = int(dm_concat)
        self.dm_tag_count = int(dm_tag_count)
        if dm and dm_concat:
            self.layer1_size = (dm_tag_count + (2 * window)) * vector_size

        self.vector_size = vector_size
        self.dv = dv or KeyedVectors(self.vector_size, mapfile_path=dv_mapfile)
        # EXPERIMENTAL lockf feature; create minimal no-op lockf arrays (1 element of 1.0)
        # advanced users should directly resize/adjust as desired after any vocab growth
        self.dv.vectors_lockf = np.ones(1, dtype=REAL)  # 0.0 values suppress word-backprop-updates; 1.0 allows

        super(Doc2Vec, self).__init__(
            sentences=corpus_iterable,
            corpus_file=corpus_file,
            vector_size=self.vector_size,
            sg=(1+dm)%2,
            null_word=self.dm_concat,
            callbacks=callbacks,
            window=window,
            epochs=epochs,
            shrink_windows=shrink_windows,
            **kwargs
        )

    def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """Update distributed bag of words model ("PV-DBOW") by training on a single document.

        Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
        :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from :mod:`gensim.models.doc2vec_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            The model to train.
        doc_words : list of str
            The input document as a list of words to be used for training. Each word will be looked up in
            the model's vocabulary.
        doctag_indexes : list of int
            Indices into `doctag_vectors` used to obtain the tags of the document.
        alpha : float
            Learning rate.
        work : np.ndarray
            Private working memory for each worker.
        train_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_doctags : bool, optional
            Whether the tag vectors should be updated.
        learn_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_hidden : bool, optional
            Whether or not the weights of the hidden layer will be updated.
        word_vectors : object, optional
            UNUSED.
        word_locks : object, optional
            UNUSED.
        doctag_vectors : list of list of float, optional
            Vector representations of the tags. If None, these will be retrieved from the model.
        doctag_locks : list of float, optional
            The lock factors for each tag.

        Returns
        -------
        int
            Number of words in the input document.

        """
        # doctag_vectors是否为空的判断，是为了区分当前是训练模式还是预测模式
        # 为空表示训练过程，从模型中直接读入
        # 不为空是预测过程，会预先生成一个随机向量传入
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
        # 这里复用的是word2vec中train_batch_sg方法，原理是通过当前词来预测上下文
        # 但是对于Docvec模型来说，当前词就是当前的paragraph vector，上下文就是段落中的每一个词
        # 因此context_vectors指定为当前的paragraph vector
        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(
                    model, word, doctag_index, alpha, learn_vectors=learn_doctags, learn_hidden=learn_hidden,
                    context_vectors=doctag_vectors, context_locks=doctag_locks
                )

        return len(doc_words)


    def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """
        Update distributed bag of words model ("PV-DBOW") by training on a single document.

        Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
        :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from :mod:`gensim.models.doc2vec_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            The model to train.
        doc_words : list of str
            The input document as a list of words to be used for training. Each word will be looked up in
            the model's vocabulary.
        doctag_indexes : list of int
            Indices into `doctag_vectors` used to obtain the tags of the document.
        alpha : float
            Learning rate.
        work : np.ndarray
            Private working memory for each worker.
        train_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_doctags : bool, optional
            Whether the tag vectors should be updated.
        learn_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_hidden : bool, optional
            Whether or not the weights of the hidden layer will be updated.
        word_vectors : object, optional
            UNUSED.
        word_locks : object, optional
            UNUSED.
        doctag_vectors : list of list of float, optional
            Vector representations of the tags. If None, these will be retrieved from the model.
        doctag_locks : list of float, optional
            The lock factors for each tag.

        Returns
        -------
        int
            Number of words in the input document.
        """

        # doctag_vectors是否为空的判断，是为了区分当前是训练模式还是预测模式
        # 为空表示训练过程，从模型中直接读入
        # 不为空是预测过程，会预先生成一个随机向量传入
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf
        # 这里复用的是word2vec中train_batch_sg方法，原理是通过当前词来预测上下文
        # 但是对于Docvec模型来说，当前词就是当前的paragraph vector，上下文就是段落中的每一个词
        # 因此context_vectors指定为当前的paragraph vector
        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(
                    model, word, doctag_index, alpha, learn_vectors=learn_doctags, learn_hidden=learn_hidden,
                    context_vectors=doctag_vectors, context_locks=doctag_locks
                )

        return len(doc_words)

    def infer_vector(self, doc_words, alpha=None, min_alpha=None, epochs=None, steps=None):
        """Infer a vector for given post-bulk training document.

        Notes
        -----
        Subsequent calls to this function may infer different representations for the same document.
        For a more stable representation, increase the number of steps to assert a stricket convergence.

        Parameters
        ----------
        doc_words : list of str
            A document for which the vector representation will be inferred.
            预测的doc，是一个string类型的list
        alpha : float, optional
            The initial learning rate. If unspecified, value from model initialization will be reused.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` over all inference epochs. If unspecified,
            value from model initialization will be reused.
        epochs : int, optional
            Number of times to train the new document. Larger values take more time, but may improve
            quality and run-to-run stability of inferred vectors. If unspecified, the `epochs` value
            from model initialization will be reused.
        steps : int, optional, deprecated
            Previous name for `epochs`, still available for now for backward compatibility: if
            `epochs` is unspecified but `steps` is, the `steps` value will be used.

        Returns
        -------
        np.ndarray
            The inferred paragraph vector for the new document.

        """
        if isinstance(doc_words, string_types):
            raise TypeError("Parameter doc_words of infer_vector() must be a list of strings (not a single string).")

        alpha = alpha or self.alpha
        min_alpha = min_alpha or self.min_alpha
        epochs = epochs or steps or self.epochs
        # 给一个新的doc生成一个随机的向量
        doctag_vectors, doctag_locks = self.trainables.get_doctag_trainables(doc_words, self.docvecs.vector_size)
        doctag_indexes = [0]
        work = zeros(self.trainables.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)

        alpha_delta = (alpha - min_alpha) / max(epochs - 1, 1)
        # 根据参数选择对应的模型：DM/DM-CONCAT/DBOW
        for i in range(epochs):
            # 预测的过程中，固定词向量和隐藏单元不更新，只更新doc的向量：doctag_vectors
            if self.sg:
                train_document_dbow(
                    self, doc_words, doctag_indexes, alpha, work,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            # neu1参数目前是没有用的：unused
            elif self.dm_concat:
                train_document_dm_concat(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            else:
                train_document_dm(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            alpha -= alpha_delta
        # 返回更新完成的paragraph vector
        return doctag_vectors[0]
