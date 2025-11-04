import math

import numpy as np
import torch
import torch.autograd.functional as functional
import time


class RiemannianAnalyzer:
  def __init__(self, Zdim, model, device):
          self.Zdim = Zdim
          self.device = device
          self.network = model.to(device)
          self.encoder = self.network.getEncoder()
          self.decoder = self.network.getDecoder()

  def generate(self, batch = 1):
      with torch.no_grad():
        ret = self.network.generate(batch_size = batch)
      return ret

  def decoderEval(self, z) :
      with torch.no_grad():
        ret = self.decoder(z.to(self.device)).to(self.device)
      return ret

  def encoderEval(self, x) :
      with torch.no_grad():
        mean_eval, logvar_eval = self.encoder(x)
        ret = self.samplingInZ(mean_eval, logvar_eval)
      return ret

  def computeJacobian(self, net, eval_point):
      global ret
      if(net == "encoder"):
        ret = functional.jacobian(func = self.encoder, inputs = eval_point.to(self.device)).to(self.device)
      elif(net == "decoder"):
        ret = functional.jacobian(func = self.decoder, inputs = eval_point.to(self.device)).to(self.device)
      else :
        print("Error: not a valid choice for the neural network part")
      return ret.cpu()

  def samplingInZ(self, mean, logvar):
      eps = torch.randn(mean.shape).to(self.device)
      sigma = 0.5 * torch.exp(logvar)
      return mean + eps * sigma

  def toNormalJacMatrix(self, temp_jac, input_size, output_size, batch_size = 1):
      if (batch_size == 1):
        ret = temp_jac.squeeze(0).squeeze(0).squeeze(2).reshape(output_size, input_size)
      else:
        ret = np.zeros(shape=(batch_size,output_size, input_size))
        for i in range(batch_size):
          ret[i, :, :] = temp_jac[i, :, :, :, i, :].squeeze(0).squeeze(0).squeeze(2).reshape(output_size, input_size)
      return ret

  def flattenImgTensor(self, x, img_dim_x, img_dim_y):
      ret = x. reshape(img_dim_x*img_dim_y, 1)
      return ret
  def unflattenImgTensor(self, x, img_dim_x, img_dim_y):
      ret = x.reshape(1,1,2,2)
      return ret


class GeodesicHandler_2Points(RiemannianAnalyzer):

  # T is the geodesic maximum index of point, which means the number of points is T+1
  def __init__(self, Zdim, T, z_start, z_end):
      super(GeodesicHandler_2Points, self).__init__(Zdim)
      self.T = T
      self.z = torch.zeros(size=(self.Zdim, self.T + 1))

      #building segment between z_start, z_end
      z0 = z_start.reshape(self.Zdim)
      zT = z_end.reshape(self.Zdim)
      for i in range(0,self.T+1):
        self.z[:, i] = z0 + ((i)/self.T) * (zT - z0)

      #storing segment
      self.segment = torch.clone(self.z)

      #just initialized
      self.alpha_max = 0.0

  def getPath(self):
      return self.z
  def getSegment(self):
    return self.segment

  def getPathAsImages(self, key):
      images = torch.zeros(size=(self.T + 1, 1, 28, 28))
      for i in range(self.T + 1):
        if (key == "geodesic") :
          images[i, :, :, :] = self.decoderEval(self.z[:,i].reshape(1,self.Zdim)).reshape(1,28,28)
        elif (key == "segment") :
          images[i, :, :, :] = self.decoderEval(self.segment[:,i].reshape(1,self.Zdim)).reshape(1,28,28)
        else :
          pass
      return images

  def computeLenght(self, key = "all"):
    if (key == "all"):
      segment_lenght, geodesic_lenght = 0.0, 0.0
      for k in range(self.T):
        segment_lenght += torch.norm((self.decoderEval(self.segment[:, k].reshape(1, self.Zdim)) - self.decoderEval(self.segment[:, k + 1].reshape(1, self.Zdim))), 2)
        geodesic_lenght += torch.norm((self.decoderEval(self.z[:, k].reshape(1, self.Zdim)) - self.decoderEval(self.z[:, k + 1].reshape(1, self.Zdim))), 2)
      print("segment lenght = [%.12f], geodesic lenght = [%.12f]" % (segment_lenght, geodesic_lenght))
    elif (key == "segment"):
      segment_lenght = 0.0
      for k in range(self.T):
        segment_lenght += torch.norm((self.decoderEval(self.segment[:, k].reshape(1, self.Zdim)) - self.decoderEval(self.segment[:, k + 1].reshape(1, self.Zdim))), 2)
      print("segment lenght = [%.12f]" % (segment_lenght))
    elif(key == "geodesic"):
      geodesic_lenght = 0.0
      for k in range(self.T):
        geodesic_lenght += torch.norm((self.decoderEval(self.z[:, k].reshape(1, self.Zdim)) - self.decoderEval(self.z[:, k + 1].reshape(1, self.Zdim))), 2)
      print("geodesic lenght = [%.12f]" % (geodesic_lenght))

  def compute_nabla_zi_E(self, index_i):
      if (index_i <= 0 or index_i >= self.T):
        print("Error in index, it must be between 1 and T-1 included!")
        return torch.zeros(size = (self.Zdim, 1))

      raw_jac_decoder_zi = self.computeJacobian("decoder", self.z[:,index_i].reshape(1,self.Zdim))
      jac_decoder_zi = self.toNormalJacMatrix(raw_jac_decoder_zi, self.Zdim, 784)

      decoder_zi_meno = self.flattenImgTensor(self.decoderEval(self.z[:,index_i-1].reshape(1,self.Zdim)), 28, 28)
      decoder_zi = self.flattenImgTensor(self.decoderEval(self.z[:,index_i].reshape(1,self.Zdim)), 28, 28)
      decoder_zi_piu = self.flattenImgTensor(self.decoderEval(self.z[:,index_i+1].reshape(1,self.Zdim)), 28, 28)

      nabla_zi_E = (
                    - (self.T) * torch.t(jac_decoder_zi.to(self.device)) @ (decoder_zi_piu.to(self.device) - 2*decoder_zi.to(self.device) + decoder_zi_meno.to(self.device))
                   ).to(self.device)

      return nabla_zi_E

  # restituisce una matrice che ha per colonne i nabla_zi_E
  def compute_nabla_E(self):
    nabla_E = torch.zeros(size = (16,self.T + 1))
    for i in range(1, self.T):
      nabla_E[:, i] = self.compute_nabla_zi_E(index_i = i).reshape(16)

    return nabla_E

  def adjust_lr(self, alpha, iter, decay_rate = 0.5, alpha_min = 0, iter_limit = 20, verbose = False):
    if (iter == 1 or iter % iter_limit == 0):
      if (iter % iter_limit == 0):
          alpha = decay_rate * self.alpha_max
      self.alpha_max = alpha
    else:
      alpha = alpha_min + 1/2 * (self.alpha_max - alpha_min) * (1 + np.cos((iter / iter_limit) * math.pi))

    if (verbose):
      print('Next learning rate: [%.12f]' %  alpha)
    return alpha

  # toll = epsilon di tolleranza sulla norma di nablaE
  # alpha_desc = alpha del metodo di discesa
  def computeGeodesic (self, toll = 0.1, alpha_desc = 0.001, max_iter = 1000, verbose = True, verbose_module = 1):
    nabla_E = self.compute_nabla_E()
    nabla_E_norm = 1
    n_iter = 0
    while nabla_E_norm > toll and n_iter < max_iter:
      start = time.time()
      for i in range(1,self.T):
        self.z[:, i] = self.z[:, i] - alpha_desc * nabla_E[:, i]
      nabla_E = self.compute_nabla_E()
      nabla_E_norm = torch.norm(nabla_E)
      n_iter += 1
      isIterToLog = n_iter % verbose_module == 0 or n_iter == 1
      if (verbose and isIterToLog):
        print ("Iteration number: %d, nabla_E norm = [%.9f]; step time = %.5f s" % (n_iter, nabla_E_norm, time.time() - start))
      alpha_desc = self.adjust_lr(alpha_desc, n_iter, verbose = isIterToLog)
