import torch 
import matplotlib.pyplot as plt
import numpy as np
import sys 
import logging


def extraElements(batchShape, device, norm, oneElement = False)-> torch.Tensor:

    
    r'''  create extra elements for be concatened in the batch tensor
            only works when batch is normlized, because it takes values for represent
            this special elements outside of [0,1] range, where is the normal elements domain
    '''
    a,b,_ = batchShape
    elementsToGenerate = 2 # default value

    if oneElement:
        elementsToGenerate = 1

    extraEle = torch.ones((a,b,elementsToGenerate), dtype = torch.float32, device = device)

    extraEle[:,:,0] = 1.5

    if not oneElement:
        extraEle[:,:,1] = 2

    return extraEle



def normalization(batch: "Torch tensor", device:"Device" = "cpu")  -> "Torch tensor with same dimention":
    r'''
    Normalize function -> (0,1)
    using the next formula by element and caracteristic
        element - minimum
        -----------------
        maximum - minimum

    this works with tensor arrange of the next way
        [batch index, caracteristic, elements]


    Probar: el normalizado entre un rango fijo de valores donde no se tome el maximo y el minomo de cada 
    instancia para tenerlos como litmires naturales del problema

    '''


    batchNorm = torch.zeros_like(batch) # default device of input
    for i, element in enumerate(batch):
        for j, parameter in enumerate(element):
            mini = min(parameter)
            maxi = max(parameter)
            batchNorm[i][j] =  (parameter - mini) / (maxi - mini)

    return batchNorm


#labels for plotting
def labelsFunc(point, mode):
    n = point['n_position']
    

    labels = []
    if mode == 'n':
       
        # labels for plotting
        for i in range(2, n + 1):
            labels.append( (i, n.item() ) )

    elif mode == 'k_n':
        siz = point['batchQntClouds']
        cloud_qnts = torch.arange(start = 2, end = siz + 1, device = 'cpu')
        ks = [torch.arange(start = 2, end = cloud_qnt + 1, device = 'cpu') for cloud_qnt in cloud_qnts]
        
        # tuple values for x's tick for matplotlib
        labels = []
        for c, b in zip(cloud_qnts,ks):
            for z in b:
                labels.append( (z.item(), c.item()) )
    
    elif mode =='k':
        siz = point['batchQntClouds']
        k = point['k_position']
        
        labels = [(k.item(), i) for i in range(k,siz+1)]

    else:
        raise RuntimeError(" arg:mode must to be 'n' or 'k_n' ")


    return labels


def plotting(args, names, pointsToGraph = None, mode:str = None,
            path = None, logScale = False, labeloptions:tuple = (False,'upper right'),
            annotatewo = True)-> "image":

    # points2graph tuple (value, name2display)
    # data_and_name

    
    legendingraph, location = labeloptions
    anotaciones = True
    
    # Validation
    log = logging.getLogger(__name__)
    log.info('Graph path: %s \n\tand inforamation\n\n', path)
    
    
        
    # Raise error
    if not(mode is None or type(mode) is str):
        raise TypeError ('Mode is getting a invalid type')

    # Validate that all data have the same length
    
    
    log.info('Values to graph:')
    to_compare = data_and_names[0][0].shape[0] #firs element of data
    for i, (data, name) in enumerate(data_and_name):
        log.info(' \tData %s:  %s\n', str(name), str(data))
        if data.shape[0] != to_compare:
            print(sys.exc_info()[0])
            raise ValueError('Tensor doesn\'t have teh same dimention')
    
    
    fig, ax = plt.subplots(figsize = (7,5.5))
    linecolor = None
    for element, name in data_and_names:
        if isinstance(element, torch.Tensor): element = element.cpu()
        if name == 'Redundancia': linecolor = '#ff7f0e'
        elif name == 'Probabilidad de perdida': linecolor = '#1f77b4'
        elif name == 'WO': linecolor = '#2ca02c'
        else: linecolor = None
        ax.plot(element, marker = 'o', label = name, color = linecolor)

        # anotations to the points

        if anotaciones and not (name == 'WO' and not annotatewo):
            for i, val in enumerate(element):
                if isinstance(val, torch.Tensor): val = val.item()
                if logScale: 
                    txt = '{:.2e}'.format(val)
                else:
                    txt = '{:.2f}'.format(val)

                if 'WO' in names:
                    if name == 'WO':
                        rotate = 90
                    else:
                        rotate = 45
                else:
                    rotate = 0
                ax.annotate(txt,
                            xy = (i,val),
                            xytext = (0,1), # One point of vertical offset
                            textcoords = 'offset points',
                            rotation= rotate,
                            ha = 'center', va = 'bottom')
    

  # necesary data for calculate position and xticks
    n = point['n_position'].float()
    k = point['k_position'].float()
    siz = point['batchQntClouds']

    if mode == 'k_n':
        position = (( (n - 2) * (n - 1) ) / 2) + k-2
        elements = (( (siz - 1) * (siz) ) / 2)
        val_ticks = torch.arange(0, elements).cpu()
    
    elif mode == 'n':
        position = k - 2
        val_ticks = torch.arange( 0, n-1 ).cpu()
    elif mode == 'k':
        position = n-k
        val_ticks = torch.arange(0,siz-k+1)
   
    
    annotations = True
    if pointsToGraph is not None and annotations:
        for (val, name) in pointsToGraph:
            log.info('\t name %s,  valor: %s', str(name), str(val) )

       
        # plotting
        for (val, name) in pointsToGraph:
            txt = 'Model'
            plt.annotate(txt,
                        xy = (position, value),
                        arrorsprops = dict(facecolor = 'black'))
        
    labels = labelFunc(point, mode)
    if labels is not None:
        plt.xticks(val_ticks, labels, rotation = 'vertical', fontsize = 8)


    title = path.split('/')[2].split('.')[0]
    ax.set(xlabel="Valores validos de la tupla (k,n)", ylabel="Valores", title=title)
    ax.grid()

    

    if logScale:
        plt.yscale("log")

    plt.rcParams.update({'font.size': 8})
    plt.subplots_adjust(bottom = 0.25, top = 0.95, right = 0.98, left = 0.1)
    # plt.axis('off')
    if legendingraph:
        plt.legend(loc = location)
    else:
        plt.legend(bbox_to_anchor=(0, -0.26), loc='center left', borderaxespad=0.)
    plt.savefig(path)
    plt.show()
    plt.close()
   
        
def main ():
    pass
    

if __name__ == '__main__':
    main()
    
     
        