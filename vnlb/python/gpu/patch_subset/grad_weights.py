

import torch,math
import torch as th
from torch import nn
from torch import optim
import numpy as np
torch.autograd.set_detect_anomaly(True)
from einops import rearrange,repeat

class WeightNet(nn.Module):

    def __init__(self,bsize,nsamples,dim):
        super(WeightNet, self).__init__()

        iweights = torch.ones(bsize,nsamples,1)
        # iweights = torch.zeros(bsize,nsamples,1)
        # iweights = 3*torch.ones(bsize,nsamples,1)
        # iweights[:,:30] = 1.

        self.relu = nn.ReLU()
        self.weights = nn.Parameter(iweights)
        # self.weights = nn.Parameter(torch.rand(bsize,nsamples,1)-0.5)
        self.bias = nn.Parameter(torch.zeros(bsize,dim))
        # self.bias = nn.Parameter(torch.zeros(bsize,nsamples))

    def forward(self,x):

        # -- compute weights [v1] --
        # weights = self.weights
        # expw = torch.exp(-weights)
        # rexp = self.relu(expw-0.5)
        # wx = expw * rexp * x

        # -- compute weights [v1.1] --
        weights = self.weights
        expw = self.relu(weights)
        # expw = torch.exp(-weights)
        # expw = weights
        wx = expw * x

        # -- compute weights [v2] --
        # rweights = self.relu(self.weights)
        # expw = torch.exp(-rweights)
        # wx = expw * x

        # -- ave patches --
        x = torch.mean(wx,dim=1)
        # x /= torch.sum(expw,dim=1)
        x = self.bias + x
        return wx,expw,x

    def freeze_bias(self):
        self.bias.requires_grad = False

    def defrost_bias(self):
        self.bias.requires_grad = True

    def freeze_weights(self):
        self.weights.requires_grad = False

    def defrost_weights(self):
        self.weights.requires_grad = True

    def get_weights(self):
        return self.weights,self.bias

    def clamp_weights(self):
        self.weights.data.clamp_(0)

    def round_weights(self,nkeep=30):
        vals,inds = torch.topk(-self.weights.data[...,0],nkeep,1)
        self.weights.data[...] = 0.
        self.weights.data[...,0].scatter_(1,inds,1.)

    def get_topk_patches(self,nkeep=30):
        vals,inds = torch.topk(-self.weights.data[...,0],nkeep,1)
        return inds

def create_optim(model,lr):
    return optim.Adam(model.parameters(),lr)

def cov_loss(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean=None):
    # return cov_loss_v1(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean)
    # return cov_loss_v2(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean)
    return cov_loss_v3(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean)

def cov_loss_v2(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean=None):
    if not(clean is None):
        rmean = clean[:,0]
    else:
        rmean = ref_patches[:,:10].mean(dim=1,keepdim=True)
    deno = wpatches.mean(dim=1,keepdim=True)
    delta = (deno - rmean)**2
    # if not(clean is None):
    #     print(deno.shape,rmean.shape,clean.shape)
    # delta = torch.mean(delta,1)
    # delta = -10*torch.log(1./delta)
    # delta = (pmean[:,None,:] - rmean)#/255.
    # delta = (wpatches - rmean)#/255.
    loss = torch.abs(torch.sum(delta) - sigma**2)
    return loss

def cov_loss_v3(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean=None):
    """
    "MMD" based loss
    """

    # -- zero mean --
    if not(clean is None):
        ref_patches = clean[:,:1]
        rmean = torch.zeros(ref_patches.shape,device=ref_patches.device)
    else:
        ref_patches = ref_patches
        rmean = ref_patches[:,:10].mean(dim=1,keepdim=True)
        # rmean = ref_patches.mean(dim=2,keepdim=True)

    # -- zero mean --
    # zm_patches = ref_patches - rmean
    # zm_patches = zm_patches.transpose(2,1)
    # sigma2 = sigma**2

    # -- bayes filter --
    # cov = th_cov(wpatches,pmean)
    # num,pdim,pdim = cov.shape
    # diag = torch.arange(pdim)
    # deno_patches = torch.linalg.solve(cov,zm_patches)
    # cov_n = cov - cov_gt
    # deno_patches = cov_n @ deno_patches
    # deno_patches = deno_patches.transpose(2,1)
    # deno_patches = deno_patches + rmean

    # -- weighted ave --
    # deno_patches = wpatches.mean(dim=1,keepdim=True)
    deno_patches = wpatches

    # -- final covariance for loss --
    res_patches = ref_patches[:,[0]] - deno_patches
    # res_cov = th_cov(res_patches,th.zeros_like(pmean))
    # res_cov = th_cov(res_patches,rmean[:,0])
    # res_cov = th_cov(res_patches,torch.zeros_like(rmean[:,0]))
    # cov_res = cov_n - cov_n @ torch.linalg.solve(cov,cov_n.transpose(2,1))
    # loss = torch.mean((res_cov - cov_res)**2)#/(255.**2)
    # loss = torch.mean((res_cov - cov_gt)**2)#/(255.**2)

    # -- gaussian patches --
    device = ref_patches.device
    bsize,num,dim = ref_patches.shape


    # -- kernels --
    loss = 0
    nsamples = 10
    g_sigma = math.sqrt(2) * sigma
    for i in range(nsamples):
        g_patches = torch.normal(0,g_sigma,size=(bsize,num,dim),device=device)
        # y_patches = torch.normal(0,sigma,size=(bsize,num,dim),device=device)
        # dists = torch.cdist(y_patches, g_patches,p=2.0)
        dists = torch.cdist(ref_patches, g_patches,p=2.0)
        mmd_sigma = torch.mean(dists[:,:100],dim=(1,2)).detach()
        # mmd_vals = mmd(y_patches, g_patches, mmd_sigma)
        mmd_vals = mmd(res_patches, g_patches, mmd_sigma)
        loss += mmd_vals.mean()
    loss /= nsamples
    # loss = -1 * loss

    return loss

def mmd(x, y, sigma):
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value,
    # pass none to not get p-value
    device = x.device
    b, n, d = x.shape
    b, m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x, y], dim=1)
    # xy = torch.cat([x.detach(), y.detach()], dim=0)
    # print("x.shape: ",x.shape)
    # print("y.shape: ",y.shape)
    # print("xy.shape: ",xy.shape)
    dists = torch.cdist(xy, xy, p=2.0)
    # print("dists.shape: ",dists.shape)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    # print("sigma.shape: ",sigma.shape)
    dists = dists**2./(2*sigma[:,None,None]**2)
    # print("dists.shape: ",dists.shape)
    k = torch.exp(-dists) + torch.eye(n+m,device=device)*1e-5
    # print("k.shape: ",k.shape)
    k_x = k[:,:n, :n]
    k_y = k[:,n:, n:]
    k_xy = k[:,:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    k_x_sum = k_x.sum(dim=(1,2))
    k_y_sum = k_y.sum(dim=(1,2))
    k_xy_sum = k_xy.sum(dim=(1,2))
    mmd = k_x_sum / (n * (n - 1)) + k_y_sum / (m * (m - 1)) - 2 * k_xy_sum / (n * m)
    return mmd

def cov_loss_v1(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean=None):

    # -- zero mean --
    if not(clean is None):
        ref_patches = clean[:,:1]
        rmean = torch.zeros(ref_patches.shape,device=ref_patches.device)
    else:
        ref_patches = ref_patches[:,:20]
        rmean = ref_patches.mean(dim=1,keepdim=True)
        # rmean = ref_patches.mean(dim=2,keepdim=True)

    # -- zero mean --
    zm_patches = ref_patches - rmean
    zm_patches = zm_patches.transpose(2,1)
    sigma2 = sigma**2

    # -- bayes filter --
    cov = th_cov(wpatches,pmean)
    num,pdim,pdim = cov.shape
    diag = torch.arange(pdim)
    deno_patches = torch.linalg.solve(cov,zm_patches)
    cov_n = cov - cov_gt
    deno_patches = cov_n @ deno_patches
    deno_patches = deno_patches.transpose(2,1)
    deno_patches = deno_patches + rmean

    # -- mean filter --
    # deno_patches = torch.mean(zm_patches,2,keepdim=True)
    # print("deno_patches.shape: ",deno_patches.shape)
    # deno_patches = deno_patches.transpose(2,1)

    # -- final covariance for loss --
    res_patches = ref_patches - deno_patches
    # res_cov = th_cov(res_patches,th.zeros_like(pmean))
    res_cov = th_cov(res_patches,rmean[:,0])
    cov_res = cov_n - cov_n @ torch.linalg.solve(cov,cov_n.transpose(2,1))
    loss = torch.mean((res_cov - cov_res)**2)#/(255.**2)
    # loss = torch.mean((res_cov - cov_gt[None,:])**2)#/(255.**2)

    # -- zero mean loss --
    # loss += torch.mean((deno_patches/255.)**2)

    return loss

def l1_reg_loss(model):
    weights,bias = model.get_weights()
    return torch.sum(torch.exp(-weights))

def th_cov(patches,mean):

    # -- cov mats --
    B,N,P = patches.shape

    # -- zero mean --
    zm_patches = patches - mean[:,None,:]
    # zm_patches = patches - mean[:,:,None]
    # print("patches.shape: ",patches.shape)
    # print("mean.shape: ",mean.shape)

    # -- cov mat --
    zm_patchesT = zm_patches.transpose(2,1)
    cov = zm_patchesT @ zm_patches / N

    return cov

# def create_grad_weights(ref_patches,candidate_patches,sigma,nkeep=100):

#     # -- shape --
#     B,N,D = ref_patches.shape
#     ref_weights = torch.ones(ref_patches.shape)



def train_cov(model,sgd_optim,ref_patches,candidate_patches,
              sigma,cov_gt,niters=100,clean=None):
    #
    # -- training candidate patches --
    #

    # -- freeze weights --
    model.defrost_weights()
    model.freeze_bias()
    prev_loss = float("inf")

    # -- update --
    w_clamp,w_round,log_step = 2,10000,5
    for i in range(niters):

        # -- forward --
        model.zero_grad()
        wpatches,weights,pmean = model(candidate_patches)
        # loss = torch.sum(torch.abs(pmean))
        loss = cov_loss(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean)
        # loss += 100*l1_reg_loss(model)

        # -- break if needed --
        # if prev_loss < loss.item():
        #     print("loss:%2.3e" % loss.item())
        #     break

        # -- update --
        loss.backward()

        for j in range(20):
            sgd_optim.step()
            model.clamp_weights()

        # -- inform --
        # print(f"[{i}/{niters}]: ",loss.item())#,l1_reg_loss(model).item())
        # if (i % w_clamp) == 0:
        #     model.clamp_weights()

        # if (i % w_round) == 0:
        #     model.round_weights()

        if (i % log_step) == 0:
            # print(wpatches[0])
            # print(weights[0].transpose(1,0))
            print("loss: %2.3e" % loss.item())
            prev_loss = loss.item()
            if not(clean is None):
                psnrs = compute_psnr_of_ave(clean[:,:1],wpatches.detach())
                print("psnrs: ",psnrs)


def compute_psnr_of_ave(cpatch,wpatches):

    # -- init --
    eps = 1e-8
    bsize,num,pdim = wpatches.shape

    # -- only 0th index --
    deno = wpatches.mean(dim=1,keepdim=True)
    delta = (cpatch - deno)**2
    delta = rearrange(delta[:,[0]],'b n p -> (b n) p')
    delta = delta.cpu().numpy()
    delta = np.mean(delta,axis=1) + eps
    log_mse = np.ma.log10(1./delta).filled(-np.infty)
    psnrs = 10 * log_mse
    psnrs = rearrange(psnrs,'(b n) -> b n',b=bsize)
    psnrs = psnrs[:,0]

    return psnrs

def train_bias(model,sgd_optim,ref_patches,candidate_patches,cov_gt,niters=100):
    #
    # -- training bias term --
    #

    # -- switch params --
    model.defrost_bias()
    model.freeze_weights()

    # -- fwd --
    pmean = model(candidate_patches)
    loss = cov_loss(ref_patches,pmean,cov_gt)
    loss.backward()

    # -- update --
    for i in range(niters):
        sgd_optim.step()

        # -- inform --
        # print(f"[{i}/{niters}]: ",loss.item())

    # print(loss.item())

def create_grad_weights(ref_patches,candidate_patches,sigma,nkeep=100,clean=None):
    return create_grad_weights_v1(ref_patches,candidate_patches,sigma,nkeep,clean)
    # return create_grad_weights_v2(ref_patches,candidate_patches,sigma,nkeep,clean)

def create_grad_weights_v1(ref_patches,candidate_patches,sigma,nkeep=100,clean=None):

    # -- shape --
    device = candidate_patches.device
    bsize,nsamples,dim = candidate_patches.shape
    sigma = sigma / 255.
    cov_gt = sigma**2 * torch.eye(dim,device=device)

    # -- center --
    ref_patches = ref_patches.clone()
    candidate_patches = candidate_patches.clone()
    ref_patches /= 255.
    candidate_patches /= 255.
    if not(clean is None):
        clean = clean.clone()
        clean /= 255.
        psnrs = compute_psnr_of_ave(clean[:,:1],ref_patches)

    # -- create model --
    model = WeightNet(bsize,nsamples,dim).to(device)

    # -- clean loss
    # if not(clean is None):
    #     cw = torch.ones_like(model.weights.data)
    #     # cw = cw[:,
    #     # loss = cov_loss(ref_patches,weights,wpatches,pmean,sigma,cov_gt)


    # -- compute loss --
    niters = 1000
    base_lr = 1e-4
    for j in range(1):

        lr = base_lr / math.sqrt((2*(j+1)))
        sgd_optim = create_optim(model,lr)
        is_break = train_cov(model,sgd_optim,ref_patches,candidate_patches,
                             sigma,cov_gt,niters=niters,clean=clean)
        # print(model.weights[0].ravel())
        if is_break: break
        # pmean = model(candidate_patches)
        # loss = cov_loss(ref_patches,pmean,cov_gt).detach()
        # print(loss.item())
        # break


        # sgd_optim = create_optim(model,1e-6)
        # is_break = train_bias(model,sgd_optim,ref_patches,candidate_patches,
        #                       cov_gt,niters=20)
         # if is_break: break

        # pmean = model(candidate_patches)
        # loss = cov_loss(ref_patches,pmean,cov_gt).detach()
        # print(loss.item())

    #
    # -- wrap it up --
    #
    model.freeze_weights()
    # model.weights[...] = 0

    nkeep = min(nkeep,nsamples)
    order = model.get_topk_patches(nkeep)
    # order = repeat(torch.arange(nkeep),'k -> b k',b=bsize)
    # order = order.to(device)
    # print(order[0])
    weights,bias = model.get_weights()
    weights = weights[:,:,0]

    # -- detach --
    order = order.detach()
    weights = weights.detach()
    bias = bias.detach()

    # -- relu --
    # weights = model.relu(weights).detach()
    # nzi = torch.where(weights > 0)
    # zi = torch.where(weights <= 0)
    # weights[nzi] = 1.
    # weights[zi] = 0.

    # -- info --
    inds = order
    print("inds.shape: ",inds.shape)
    print("weights.shape ",weights.shape)

    # -- select v1 --
    nkeep = 100
    weights[...] = 0
    weights = weights.scatter(1,inds[:,:nkeep],1.)

    # -- other select --
    # weights = torch.exp(-weights)
    # rweights = model.relu(weights-0.5)
    # weights = weights * rweights

    # weights = torch.exp(-weights)
    # weights /= torch.sum(weights,1,keepdim=True)
    # weights = model.relu(weights-0.5)+0.5

    return order,weights,bias

def create_grad_weights_v2(ref_patches,candidate_patches,sigma,nkeep=100,clean=None):


    # -- init --
    color = 3
    device = candidate_patches.device
    bsize,nsamples,dim = candidate_patches.shape

    # -- normalize --

    ref_patches = ref_patches.clone()
    candidate_patches = candidate_patches.clone()
    ref_patches /= 255.
    candidate_patches /= 255.
    if not(clean is None):
        clean = clean.clone()
        clean /= 255.
    sigma = sigma/255.

    # -- create vars --
    cov_gt = (sigma)**2 * torch.eye(dim,device=device)
    weights = torch.ones(bsize,nsamples,1,device=device)

    # -- forward --
    weights.requires_grad = True
    wpatches = weights * candidate_patches
    pmean = wpatches.mean(dim=1)

    # -- gradient --
    print(ref_patches.max())
    print(candidate_patches.max())
    print(sigma,sigma*255.)
    loss = cov_loss(ref_patches,weights,wpatches,pmean,sigma,cov_gt,clean)
    grad = torch.autograd.grad(loss,weights)[0].detach()
    # grad = torch.abs(grad)
    # grad = rearrange(grad,'(b c) p 1 -> b c p',c=3)
    # grad = torch.mean(grad,1)
    # grad = torch.rand_like(grad)
    # grad = torch.rand(bsize,nsamples,1,device=device)
    # grad[...] = 0.
    # grad[:,:100] = 1.

    # -- get indices to mark as "1" --
    topK = min(100,nsamples)
    grad[:,:3] = 100000.
    grad = rearrange(grad,'(b c) n 1 -> b c n',c=3)
    grad = torch.mean(grad,1)
    inds = torch.argsort(grad,1)
    # vals,inds = torch.topk(grad,topK,1)
    # print(grad[0].ravel())
    # print("inds.shape: ",inds.shape)
    # print(inds[0].ravel())

    # -- apply to weights --
    nkeep = 100
    weights = weights.detach()

    # -- select v1 --
    weights[...] = 0
    weights = weights[:,:,0]
    weights = weights.scatter(1,inds[:,:nkeep],1.)

    # -- select v2 --
    # weights = rearrange(weights,'(b c) p 1 -> b c p',c=3)
    # weights[...] = 0.
    # for ci in range(color):
    #     weights[:,ci] = weights[:,ci].scatter(1,inds[:,:nkeep],1.)
    # weights = rearrange(weights,'b c p -> (b c) p',c=3)

    # -- misc --
    order = inds
    bias = 0.

    return order,weights,bias
