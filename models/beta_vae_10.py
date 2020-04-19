import logging
import os

import keras
import numpy
from keras import backend as K, Model
from keras.callbacks import CSVLogger
from keras.callbacks import LambdaCallback,EarlyStopping,ModelCheckpoint
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, Lambda
from keras.models import load_model
from scipy import sparse

#import scgen
import util_lossC as ul
#from util_lossC import balancer,extractor,shuffle_adata

log = logging.getLogger(__file__)



class C_VAEArithKeras:
    """
        VAE with Arithmetic vector Network class. This class contains the implementation of Variational
        Auto-encoder network with Vector Arithmetics.
        Parameters
        ----------
        kwargs:
            :key `validation_data` : AnnData
                must be fed if `use_validation` is true.
            :key dropout_rate: float
                    dropout rate
            :key learning_rate: float
                learning rate of optimization algorithm
            :key model_path: basestring
                path to save the model after training
        x_dimension: integer
            number of gene expression space dimensions.
        z_dimension: integer
            number of latent space dimensions.
        See also
        --------
        CVAE from scgen.models._cvae : Conditional VAE implementation.
    """

    def __init__(self, x_dimension, z_dimension=100 , **kwargs):
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.dropout_rate = kwargs.get("dropout_rate", 0.2)
        self.model_to_use = kwargs.get("model_to_use", "./models/")
        self.alpha = kwargs.get("alpha", 0.00005)
        self.c_max = kwargs.get("c_max", 20)
        self.c_current = K.variable(value=0.01)
        self.x = Input(shape=(x_dimension,), name="input")
        self.z = Input(shape=(z_dimension,), name="latent")
        self.init_w = keras.initializers.glorot_normal()
        self._create_network()
        self._loss_function()
        self.vae_model.summary()

    def _encoder(self):
        """
            Constructs the encoder sub-network of VAE. This function implements the
            encoder part of Variational Auto-encoder. It will transform primary
            data in the `n_vars` dimension-space to the `z_dimension` latent space.
            Parameters
            ----------
            No parameters are needed.
            Returns
            -------
            mean: Tensor
                A dense layer consists of means of gaussian distributions of latent space dimensions.
            log_var: Tensor
                A dense layer consists of log transformed variances of gaussian distributions of latent space dimensions.
        """
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(self.x)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)
        # h = Dense(512, kernel_initializer=self.init_w, use_bias=False)(h)
        # h = BatchNormalization()(h)
        # h = LeakyReLU()(h)
        # h = Dropout(self.dropout_rate)(h)
        # h = Dense(256, kernel_initializer=self.init_w, use_bias=False)(h)
        # h = BatchNormalization()(h)
        # h = LeakyReLU()(h)
        # h = Dropout(self.dropout_rate)(h)

        mean = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        log_var = Dense(self.z_dim, kernel_initializer=self.init_w)(h)
        z = Lambda(self._sample_z, output_shape=(self.z_dim,), name="Z")([mean, log_var])

        self.encoder_model = Model(inputs=self.x, outputs=z, name="encoder")
        return mean, log_var

    def _decoder(self):
        """
            Constructs the decoder sub-network of VAE. This function implements the
            decoder part of Variational Auto-encoder. It will transform constructed
            latent space to the previous space of data with n_dimensions = n_vars.
            Parameters
            ----------
            No parameters are needed.
            Returns
            -------
            h: Tensor
                A Tensor for last dense layer with the shape of [n_vars, ] to reconstruct data.
        """
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(self.z)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)
        h = Dense(800, kernel_initializer=self.init_w, use_bias=False)(h)
        h = BatchNormalization(axis=1)(h)
        h = LeakyReLU()(h)
        h = Dropout(self.dropout_rate)(h)
        # h = Dense(768, kernel_initializer=self.init_w, use_bias=False)(h)
        # h = BatchNormalization()(h)
        # h = LeakyReLU()(h)
        # h = Dropout(self.dropout_rate)(h)
        # h = Dense(1024, kernel_initializer=self.init_w, use_bias=False)(h)
        # h = BatchNormalization()(h)
        # h = LeakyReLU()(h)
        # h = Dropout(self.dropout_rate)(h)
        h = Dense(self.x_dim, kernel_initializer=self.init_w, use_bias=True)(h)

        self.decoder_model = Model(inputs=self.z, outputs=h, name="decoder")
        return h

    @staticmethod
    def _sample_z(args):
        """
            Samples from standard Normal distribution with shape [size, z_dim] and
            applies re-parametrization trick. It is actually sampling from latent
            space distributions with N(mu, var) computed in `_encoder` function.
            Parameters
            ----------
            No parameters are needed.
            Returns
            -------
            The computed Tensor of samples with shape [size, z_dim].
        """
        mu, log_var = args
        batch_size = K.shape(mu)[0]
        z_dim = K.shape(mu)[1]
        eps = K.random_normal(shape=[batch_size, z_dim])
        return mu + K.exp(log_var / 2) * eps

    def _create_network(self):
        """
            Constructs the whole VAE network. It is step-by-step constructing the VAE
            network. First, It will construct the encoder part and get mu, log_var of
            latent space. Second, It will sample from the latent space to feed the
            decoder part in next step. Finally, It will reconstruct the data by
            constructing decoder part of VAE.
            Parameters
            ----------
            No parameters are needed.
            Returns
            -------
            Nothing will be returned.
        """
        self.mu, self.log_var = self._encoder()

        self.x_hat = self._decoder()
        self.vae_model = Model(inputs=self.x, outputs=self.decoder_model(self.encoder_model(self.x)), name="VAE")

    def _loss_function(self):
        """
            Defines the loss function of VAE network after constructing the whole
            network. This will define the KL Divergence and Reconstruction loss for
            VAE and also defines the Optimization algorithm for network. The VAE Loss
            will be weighted sum of reconstruction loss and KL Divergence loss.
            Parameters
            ----------
            No parameters are needed.
            Returns
            -------
            Nothing will be returned.
        """

        def vae_loss(y_true, y_pred):
            print(self.c_current)
            return K.mean(recon_loss(y_true, y_pred) + self.alpha *abs(kl_loss(y_true, y_pred)-self.c_current))

        def kl_loss(y_true, y_pred):
            return 0.5 * K.sum(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=1)

        def kl_loss_monitor0(y_true, y_pred):
            klds =  K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            return klds[0]
        
        def kl_loss_monitor1(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            #K.print_tensor(klds)
            return klds[1]
        
        def kl_loss_monitor2(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            #K.print_tensor(klds)
            return klds[2]

        def kl_loss_monitor3(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            #K.print_tensor(klds)
            return klds[3]

        def kl_loss_monitor4(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            #K.print_tensor(klds)
            return klds[4]

        def kl_loss_monitor5(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            return klds[5]

        def kl_loss_monitor6(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            return klds[6]

        def kl_loss_monitor7(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            return klds[7]
        
        def kl_loss_monitor8(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            return klds[8]
        
        def kl_loss_monitor9(y_true, y_pred):
            klds = K.mean(K.exp(self.log_var) + K.square(self.mu) - 1. - self.log_var, axis=0)
            return klds[9]

        def recon_loss(y_true, y_pred):
            return 0.5 * K.sum(K.square((y_true - y_pred)), axis=1)
        
        def get_c_current(y_true, y_pred):
            return self.c_current

        self.vae_optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.vae_model.compile(optimizer=self.vae_optimizer, loss=vae_loss, 
        metrics=[kl_loss ,recon_loss,get_c_current,kl_loss_monitor0,kl_loss_monitor1,kl_loss_monitor2,kl_loss_monitor3,
        kl_loss_monitor4,kl_loss_monitor5,kl_loss_monitor6,kl_loss_monitor7,kl_loss_monitor8,kl_loss_monitor9])
        
    def to_latent(self, data):
        """
            Map `data` in to the latent space. This function will feed data
            in encoder part of VAE and compute the latent space coordinates
            for each sample in data.
            Parameters
            ----------
            data:  numpy nd-array
                Numpy nd-array to be mapped to latent space. `data.X` has to be in shape [n_obs, n_vars].
            Returns
            -------
            latent: numpy nd-array
                Returns array containing latent space encoding of 'data'
        """
        latent = self.encoder_model.predict(data)
        return latent

    def _avg_vector(self, data):
        """
            Computes the average of points which computed from mapping `data`
            to encoder part of VAE.
            Parameters
            ----------
            data:  numpy nd-array
                Numpy nd-array matrix to be mapped to latent space. Note that `data.X` has to be in shape [n_obs, n_vars].
            Returns
            -------
                The average of latent space mapping in numpy nd-array.
        """
        latent = self.to_latent(data)
        latent_avg = numpy.average(latent, axis=0)
        return latent_avg

    def reconstruct(self, data):
        """
            Map back the latent space encoding via the decoder.
            Parameters
            ----------
            data: `~anndata.AnnData`
                Annotated data matrix whether in latent space or gene expression space.
            use_data: bool
                This flag determines whether the `data` is already in latent space or not.
                if `True`: The `data` is in latent space (`data.X` is in shape [n_obs, z_dim]).
                if `False`: The `data` is not in latent space (`data.X` is in shape [n_obs, n_vars]).
            Returns
            -------
            rec_data: 'numpy nd-array'
                Returns 'numpy nd-array` containing reconstructed 'data' in shape [n_obs, n_vars].
        """
        rec_data = self.decoder_model.predict(x=data)
        return rec_data

    def linear_interpolation(self, source_adata, dest_adata, n_steps):
        """
            Maps `source_adata` and `dest_adata` into latent space and linearly interpolate
            `n_steps` points between them.
            Parameters
            ----------
            source_adata: `~anndata.AnnData`
                Annotated data matrix of source cells in gene expression space (`x.X` must be in shape [n_obs, n_vars])
            dest_adata: `~anndata.AnnData`
                Annotated data matrix of destinations cells in gene expression space (`y.X` must be in shape [n_obs, n_vars])
            n_steps: int
                Number of steps to interpolate points between `source_adata`, `dest_adata`.
            Returns
            -------
            interpolation: numpy nd-array
                Returns the `numpy nd-array` of interpolated points in gene expression space.
            Example
            --------
            >>> import anndata
            >>> import scgen
            >>> train_data = anndata.read("./data/train.h5ad")
            >>> validation_data = anndata.read("./data/validation.h5ad")
            >>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
            >>> network.train(train_data=train_data, use_validation=True, validation_data=validation_data, shuffle=True, n_epochs=2)
            >>> souece = train_data[((train_data.obs["cell_type"] == "CD8T") & (train_data.obs["condition"] == "control"))]
            >>> destination = train_data[((train_data.obs["cell_type"] == "CD8T") & (train_data.obs["condition"] == "stimulated"))]
            >>> interpolation = network.linear_interpolation(souece, destination, n_steps=25)
        """
        if sparse.issparse(source_adata.X):
            source_average = source_adata.X.A.mean(axis=0).reshape((1, source_adata.shape[1]))
        else:
            source_average = source_adata.X.A.mean(axis=0).reshape((1, source_adata.shape[1]))

        if sparse.issparse(dest_adata.X):
            dest_average = dest_adata.X.A.mean(axis=0).reshape((1, dest_adata.shape[1]))
        else:
            dest_average = dest_adata.X.A.mean(axis=0).reshape((1, dest_adata.shape[1]))
        start = self.to_latent(source_average)
        end = self.to_latent(dest_average)
        vectors = numpy.zeros((n_steps, start.shape[1]))
        alpha_values = numpy.linspace(0, 1, n_steps)
        for i, alpha in enumerate(alpha_values):
            vector = start * (1 - alpha) + end * alpha
            vectors[i, :] = vector
        vectors = numpy.array(vectors)
        interpolation = self.reconstruct(vectors)
        return interpolation

    def predict(self, adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None, obs_key="all"):
        """
            Predicts the cell type provided by the user in stimulated condition.
            Parameters
            ----------
            celltype_to_predict: basestring
                The cell type you want to be predicted.
            obs_key: basestring or dict
                Dictionary of celltypes you want to be observed for prediction.
            adata_to_predict: `~anndata.AnnData`
                Adata for unpertubed cells you want to be predicted.
            Returns
            -------
            predicted_cells: numpy nd-array
                `numpy nd-array` of predicted cells in primary space.
            delta: float
                Difference between stimulated and control cells in latent space
            Example
            --------
            >>> import anndata
            >>> import scgen
            >>> train_data = anndata.read("./data/train.h5ad"
            >>> validation_data = anndata.read("./data/validation.h5ad")
            >>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
            >>> network.train(train_data=train_data, use_validation=True, validation_data=validation_data, shuffle=True, n_epochs=2)
            >>> prediction, delta = pred, delta = scg.predict(adata= train_new,conditions={"ctrl": "control", "stim":"stimulated"},
                                                  cell_type_key="cell_type",condition_key="condition",adata_to_predict=unperturbed_cd4t)
        """
        if obs_key == "all":
            ctrl_x = adata[adata.obs["condition"] == conditions["ctrl"], :]
            stim_x = adata[adata.obs["condition"] == conditions["stim"], :]
            ctrl_x = ul.balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
            stim_x = ul.balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        else:
            key = list(obs_key.keys())[0]
            values = obs_key[key]
            subset = adata[adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs["condition"] == conditions["ctrl"], :]
            stim_x = subset[subset.obs["condition"] == conditions["stim"], :]
            if len(values) > 1:
                ctrl_x = ul.balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
                stim_x = ul.balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = ul.extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
        else:
            ctrl_pred = adata_to_predict
        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = numpy.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = numpy.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
            latent_ctrl = self._avg_vector(ctrl_x.X.A[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X.A[stim_ind, :])
        else:
            latent_ctrl = self._avg_vector(ctrl_x.X[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X[stim_ind, :])
        delta = latent_sim - latent_ctrl
        if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent(ctrl_pred.X.A)
        else:
            latent_cd = self.to_latent(ctrl_pred.X)
        stim_pred = delta + latent_cd
        predicted_cells = self.reconstruct(stim_pred)
        return predicted_cells, delta

    def restore_model(self):
        """K.variable(value=0.0)
            restores model weights from `model_to_use`.
            Parameters
            ----------
            No parameters are needed.
            Returns
            -------
            Nothing will be returned.
            Example
            --------
            >>> import anndata
            >>> import scgen
            >>> train_data = anndata.read("./data/train.h5ad")
            >>> validation_data = anndata.read("./data/validation.h5ad")
            >>> network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test" )
            >>> network.restore_model()
        """
        self.vae_model = load_model(os.path.join(self.model_to_use, 'vae.h5'), compile=False)
        self.encoder_model = load_model(os.path.join(self.model_to_use, 'encoder.h5'), compile=False)
        self.decoder_model = load_model(os.path.join(self.model_to_use, 'decoder.h5'), compile=False)
        self._loss_function()

    def train(self, train_data, validation_data=None,
              n_epochs=25,
              batch_size=32,
              early_stop_limit=20,
              threshold=0.0025,
              initial_run=True,
              shuffle=True,
              verbose=1,
              save=True,
              checkpoint=50,
              **kwargs):
        """
            Trains the network `n_epochs` times with given `train_data`
            and validates the model using validation_data if it was given
            in the constructor function. This function is using `early stopping`
            technique to prevent over-fitting.
            Parameters
            ----------
            train_data: scanpy AnnData
                Annotated Data Matrix for training VAE network.
            validation_data: scanpy AnnData
                Annotated Data Matrix for validating VAE network after each epoch.
            n_epochs: int
                Number of epochs to iterate and optimize network weights
            batch_size: integer
                size of each batch of training dataset to be fed to network while training.
            early_stop_limit: int
                Number of consecutive epochs in which network loss is not going lower.
                After this limit, the network will stop training.
            threshold: float
                Threshold for difference between consecutive validation loss values
                if the difference is upper than this `threshold`, this epoch will not
                considered as an epoch in early stopping.
            initial_run: bool
                if `True`: The network will initiate training and log some useful initial messages.
                if `False`: Network will resume the training using `restore_model` function in order
                    to restore last model which has been trained with some training dataset.
            shuffle: bool
                if `True`: shuffles the training dataset
            Returns
            -------
            Nothing will be returned
            Example
            --------
            ```python
            import anndata
            import scgen
            train_data = anndata.read("./data/train.h5ad"
            validation_data = anndata.read("./data/validation.h5ad"
            network = scgen.VAEArith(x_dimension= train_data.shape[1], model_path="./models/test")
            network.train(train_data=train_data, use_validation=True, valid_data=validation_data, shuffle=True, n_epochs=2)
            ```
        """
        if initial_run:
            log.info("----Training----")
        if shuffle:
            train_data = ul.shuffle_adata(train_data)

        if sparse.issparse(train_data.X):
            train_data.X = train_data.X.A


        # def on_epoch_end(epoch, logs):
        #     if epoch % checkpoint == 0:
        #         path_to_save = os.path.join(kwargs.get("path_to_save"), f"epoch_{epoch}") + "/"
        #         scgen.visualize_trained_network_results(self, vis_data, kwargs.get("cell_type"),
        #                                                 kwargs.get("conditions"),
        #                                                 kwargs.get("condition_key"), kwargs.get("cell_type_key"),
        #                                                 path_to_save,
        #                                                 plot_umap=False,
        #                                                 plot_reg=True)

        # class MyCustomCallback(keras.callbacks.Callback):
        #     def on_epoch_begin(self, epoch, logs=None):
        #         K.set_value(self.c_current, (self.c_max/n_epochs)* epoch)
        #         print("Setting C to =", str(self.c_current))
        #         print("Changed1") 

        os.makedirs(self.model_to_use, exist_ok=True)
        
        def update_val_c(epoch):
            print(epoch)
            value = (self.c_max/n_epochs)+K.get_value(self.c_current)
            K.set_value(self.c_current,value)
                    
        callbacks = [
            LambdaCallback(on_epoch_end=lambda epoch, log: update_val_c(epoch)),
            # EarlyStopping(patience=early_stop_limit, monitor='loss', min_delta=threshold),
            CSVLogger(filename=self.model_to_use+"/csv_logger.log"),
            ModelCheckpoint(os.path.join(self.model_to_use+"/model_checkpoint.h5"),monitor='vae_loss',verbose=1),
            EarlyStopping(monitor='vae_loss',patience=5,verbose=1)
        ]
        
        K.set_value(self.c_current,(self.c_max/n_epochs))

        if validation_data is not None:
            result = self.vae_model.fit(x=train_data.X,
                                        y=train_data.X,
                                        epochs=n_epochs,
                                        batch_size=batch_size,
                                        validation_data=(validation_data.X, validation_data.X),
                                        shuffle=shuffle,
                                        callbacks=callbacks,
                                        verbose=verbose)
        else:
            result = self.vae_model.fit(x=train_data.X,
                                        y=train_data.X,
                                        epochs=n_epochs,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        callbacks=callbacks,
                                        verbose=verbose)

        if save is True:
            #os.chdir(self.model_to_use)
            self.vae_model.save(os.path.join(self.model_to_use+"/vae.h5"), overwrite=True)
            self.encoder_model.save(os.path.join(self.model_to_use+"/encoder.h5"), overwrite=True)
            self.decoder_model.save(os.path.join(self.model_to_use+"/decoder.h5"), overwrite=True)
            log.info(f"Models are saved in file: {self.model_to_use}. Training finished")
        return result

