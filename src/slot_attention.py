
import torch
from torch.nn import Conv2d, ConvTranspose2d, Module, Parameter, GRU
from torch.nn import LayerNorm, Linear, Softmax, ReLU, Sequential
from torch.distributions import Normal
import numpy as np
import pdb

'''gives attention over keys and return values'''
class Attention(Module):

    def __init__(self, eps):
        super(Attention, self).__init__()

        #the output softmax
        self.eps = eps
        self.softmax = Softmax(dim = -2)

    def forward(self, keys, values, query):
        '''
        Args:
            keys: B x N x D
            values: B x N x D'
            query: B x N' x D 
        Output:
            output: B x N' x D'
        '''
        #the hidden dimension
        d = keys.shape[-1]
        
        #change dimension of keys
        #[B x 1 x N x D]
        keys = keys.unsqueeze(dim = 1)

        #[B x N' x 1 x D]
        query = query.unsqueeze(dim = 2)

        #the product
        #[B x N' x N x D]
        alpha = keys*query
        
        #take sum
        #[B x N' x N]
        alpha = torch.sum(alpha, dim = -1)/(d**0.5)

        #apply softmax
        #[B x N' x N]
        alpha = self.softmax(alpha)
        alpha = alpha + self.eps

        #wighted sum
        #[B x N' x 1]
        alpha_norm = torch.sum(alpha, dim = -1, keepdim= True)
        
        #[B x N' x N x 1]
        alpha = alpha/alpha_norm
        alpha = alpha.unsqueeze(-1)

        #final answer
        #[B x 1 x N x D]
        values = values.unsqueeze(1)
        
        #[B x N' x N x D]
        output = alpha*values
        #[B x N' x D]
        output = torch.sum(output, dim = 2)

        return output

'''positional embedding class'''
class PositionEmbed(Module):

    def __init__(self, args, input_dim, resolution):
        super(PositionEmbed, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.layer = Linear(4,input_dim)
        self.grid = self.build_grid(resolution)
    
    def build_grid(self, resolution):
        ranges = [np.linspace(0., 1., num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis=-1)
        grid = torch.Tensor(grid).to(self.args['device'])
        return grid 

    '''get the forward class'''
    def forward(self, x):
        position_embed = self.layer(self.grid)
        position_embed = position_embed.permute([0,-1,1,2])
        output = x + position_embed
        return output

'''
SlotAttention OUTPUT:
    1. B x NUM_SLOTS x SLOT_DIM
'''
class SlotAttention(Module):

    def __init__(self, args):
        
        #the super call
        super(SlotAttention,self).__init__()

        #save args
        self.args = args
        self.device = args['device']

        #number of slots
        self.number_of_slots = self.args['slot']['number_of_slots']
        
        #the input dimension and slot dimension
        self.input_dim = self.args['slot']['input_dim']
        self.slot_dim = self.args['slot']['slot_dim']
        self.hidden_dimension = self.args['slot']['hidden_dim']
        self.hidden_dimension_kv = self.args['slot']['hidden_dimension_kv']
        
        #the iterations
        self.iterations = self.args['slot']['iterations']
        self.eps = self.args['slot']['eps']

        #the slot mu and sigma
        self.mu = Parameter(torch.randn(1, 1, self.slot_dim))
        self.log_sigma = Parameter(torch.zeros(1, 1, self.slot_dim))

        #the layer and 
        self.key = Linear(self.input_dim, self.hidden_dimension_kv, bias = False)
        self.value = Linear(self.input_dim, self.slot_dim,  bias = False)
        self.query = Linear(self.input_dim, self.hidden_dimension_kv,  bias = False)
        
        #the attention module
        self.attention = Attention(self.eps)

        #we are using reparameterization technique
        self.dist = Normal(0, 1)

        #the update gru
        self.update_gru = GRU(
            input_size = self.slot_dim,
            hidden_size = self.args['slot']['gru']['hidden_dim'],
            batch_first = True
        )

        #the update mlp
        self.update_mlp = Sequential(
            Linear(self.slot_dim, self.slot_dim),
            ReLU(),
            Linear(self.slot_dim, self.slot_dim)
        )

        #the layer norm layers
        self.input_ln = LayerNorm(self.input_dim)
        self.slot_ln = LayerNorm(self.slot_dim)
        self.update_ln = LayerNorm(self.slot_dim)

    def forward(self, x):
        '''
        Args:
            x: input features with shape
                [B (H*W) C]
        Returns:
            slots: slot vectors of shape
                [B NUM_OF_SLOTS SLOT_DIM]
        '''
        #layer-norm x
        x_norm = self.input_ln(x)

        #batches
        number_of_batches = x_norm.shape[0]

        #initalize slots as
        mu = self.mu.repeat(number_of_batches, self.number_of_slots, 1)
        log_sigma = self.log_sigma.repeat(number_of_batches, self.number_of_slots, 1)

        noise = self.dist.sample((number_of_batches, self.number_of_slots, self.slot_dim)).to(self.device)
        slots = mu + (torch.exp(log_sigma))*noise
        
        #get the key and values from input
        
        #shape: [B (H*W) HIDD_DIM_KV]
        keys = self.key(x_norm) 

        #shape [B (H*W) SLOT_DIM]
        values = self.value(x_norm)
        
        #iterate over iterations
        for i in range(self.iterations):

            #set the prev slots
            prev_slots = slots
            slots = self.slot_ln(slots)

            #query_vector
            # query = self.query(prev_slots)
            query = self.query(slots)

            #slots new
            #[B x NUM_OF_SLOTS x SLOT_DIM]
            updates = self.attention(keys, values, query)

            #we need to change this to appropiate input of gru
            #[(B x NUM_OF_SLOTS) x 1 x SLOT_DIM]
            updates = updates.view(-1, 1, self.slot_dim)            
            
            #[1 x (B x NUM_OF_SLOTS) x SLOT_DIM]
            prev_slots = prev_slots.view(1, -1, self.slot_dim)

            #[1 x (B x NUM_OF_SLOTS) x SLOT_DIM]
            _, slots = self.update_gru(updates, prev_slots)
            slots = slots.view(-1, self.number_of_slots, self.slot_dim)

            #from mlp
            slots_n = self.update_ln(slots)
            slots = slots + self.update_mlp(slots_n)
        
        #return the slots
        return slots

'''
Encoder:
    It keeps image size same and at last we get
    C x H x W dimensional output
'''
class Encoder(Module):

    def __init__(self, args):
        
        #the super call
        super(Encoder,self).__init__()

        #save args
        self.args = args
        self.out_channels = self.args['encoder']['out_channels']
        self.in_channels = self.args['encoder']['in_channels']
        
        #the layers
        self.conv1 = Conv2d(self.in_channels, self.out_channels, (5,5), padding = 2)
        self.conv2 = Conv2d(self.out_channels, self.out_channels, (5,5), padding = 2, stride = 2)
        self.conv3 = Conv2d(self.out_channels, self.out_channels, (5,5), padding = 2, stride = 1) #Trying with stride 1
        self.conv4 = Conv2d(self.out_channels, self.out_channels, (5,5), padding = 2)
        
        #activations
        self.relu = ReLU()

    def forward(self, x):
        '''
        Args:
            x: [B x self.in_channels x H x W] 
        Returns:
            output: [B x self.out_channels x H x W ]
        '''
        # x = x.to(torch.float16)
        output = self.conv1(x)
        output = self.relu(output)
        
        output = self.conv2(output)
        output = self.relu(output)
        
        output = self.conv3(output)
        output = self.relu(output)
        
        output = self.conv4(output)
        output = self.relu(output)
        
        return output


'''Decoder take slots as input
with arrangement in grid and positional embedding concatednated
'''
class Decoder(Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        
        #get the args
        self.args = args

        #get the input channels
        self.in_channels = self.args['decoder']['in_channels']
        self.hidden_channels = self.args['decoder']['hidden_channels']
        self.output_channels = self.args['decoder']['out_channels']
        
        #increase the size by 16 folds
        conv_layers = [ConvTranspose2d(self.in_channels, self.hidden_channels, (5,5), (2,2), padding = 2, output_padding = 1)]
        conv_layers.append(ReLU())
        for i in range(3):
            conv_layers.append(ConvTranspose2d(self.hidden_channels, self.hidden_channels, (5,5), (2,2), padding = 2, output_padding = 1))
            conv_layers.append(ReLU())
        if(self.args['dataset'] == 'multi-object'):
            conv_layers.append(ConvTranspose2d(self.hidden_channels, self.hidden_channels, (5,5), (2,2), padding = 2, output_padding = 1))
            conv_layers.append(ReLU())

        #make it sequention
        self.conv_layers = Sequential(*conv_layers)            

        #the final two convs
        self.conv6 = ConvTranspose2d(self.hidden_channels, self.output_channels+1, (5,5), (1,1), padding = 2)
        self.conv7 = ConvTranspose2d(self.output_channels+1, self.output_channels, (3,3), (1,1), padding = 1)
                
        #activation
        self.relu = ReLU()

    def forward(self, x):
        '''
        Args:   
            x: [B x C' x H x W] tensor 
        Returns:
            output: [B x C x 16H x 16W] image
        '''
        output = self.conv_layers(x)
        output = self.conv6(output)
        output = self.relu(output)

        output = self.conv7(output)

        return output

#the slot init module
class SlotInit(Module):
    
    def __init__(self, args):
        super(SlotInit, self).__init__()
        self.args = args
        
        self.in_channels = self.args['encoder']['out_channels']
        self.in_dim = (self.args['image']['size']-4)**2
        self.out_dim = self.args['slot']['slot_dim']
        self.image_size = self.args['image']['size']

        self.relu = ReLU()
        self.conv = Conv2d(self.in_channels, 1, (5,5))
        self.linear = Linear(self.in_dim, self.out_dim)
        self.linear_m = Linear(self.out_dim, self.out_dim)
        self.linear_s = Linear(self.out_dim, self.out_dim)
    
    def forward(self, x):
        '''
        x: B H*W C
        '''
        b = x.shape[0]
        x = x.view(b,self.image_size, self.image_size, -1)
        x = torch.permute(x, (0, -1, 1, 2))
        
        out = self.conv(x)
        out = self.relu(out)
        out = out.view(-1, self.in_dim)
        out = self.linear(out)
        out = self.relu(out)
        mu = self.linear_m(out)
        sigma = self.linear_s(out)
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        return mu, sigma


'''
UOD: unsupervised object discovery
'''
class UOD(Module):

    def __init__(self, args):
        
        #the super call
        super(UOD,self).__init__()

        #copy args
        self.args = args
        self.device = args['device']
        self.encode_dim =  self.args['encoder']['out_channels']
        self.slot_dim = self.args['slot']['slot_dim']
        self.number_of_slots = self.args['slot']['number_of_slots']

        #get the models
        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)
        self.slot_attention = SlotAttention(self.args)

        #the position embeddings
        self.encoder_final_resolution = (112, 112)  
        self.decoder_inital_resolution = self.args['decoder']['initial_resolution']
        self.encoder_pos = PositionEmbed(self.args, self.encode_dim, self.encoder_final_resolution)        
        self.decoder_pos = PositionEmbed(self.args, self.slot_dim, self.decoder_inital_resolution)

        #the normalization layer
        self.encoder_ln = LayerNorm(self.slot_dim)

        #input mlp
        self.slot_mlp = Sequential(
            Linear(self.encode_dim, self.slot_dim),
            ReLU(),
            Linear(self.slot_dim, self.slot_dim)
        )

        #the final softmax layer
        self.softmax = Softmax(dim = 1)

    #just repeat the vector
    def spatial_broadcast(self, x):
        '''
        Args:
            x: input tensor of shape [B x NUM_OF_SLOTS x SLOT_DIM]
        Output:
            output: tensor of shape [(B x NUM_OF_SLOTS) x R[0] x R[1] x SLOT_DIM]
        '''
        #absorb num of slots
        output = x.view(-1, 1, 1, self.slot_dim)
        output = output.repeat(1, self.decoder_inital_resolution[0], self.decoder_inital_resolution[1], 1)
        return output
    
    '''flat the tensor'''
    def spatial_flatten(self, x):
        '''
        Args:
            x: tensor of shape [B x C x H x W]
        Returns:
            output: flatten tensor of shape
                    [B x (H x W) x C]
        '''
        output = x.permute(0,2,3,1)
        output = output.flatten(1,2)
        return output

    def forward(self, data):
        '''
        Args:
            image: [B x C x W x H]
        Output:

        '''
        #encode the image
        #[B x SLOT_DIM x W x H]
        image = data['images']
        encoding = self.encoder(image)

        #add the positional encoding
        #[B x SLOT_DIM x W x H]
        x = self.encoder_pos(encoding)
        
        #convert to [B x (W x H) x SLOT_DIM]
        x = self.spatial_flatten(x)

        #layer norm and feed to mlp
        x = self.encoder_ln(x)

        #mapping input of [B x (W x H) x SLOT_DIM]
        #to slots of [B x (W x H) x SLOT_DIM]
        x = self.slot_mlp(x)

        # Slot Attention module.
        #to slots of [B x NUM_OF_SLOTS x SLOT_DIM]
        slots = self.slot_attention(x)
        
        # Spatial broadcast decoder.
        # [(B x NUM_OF_SLOTS) x H_init x W_init x SLOT_DIM]
        x = self.spatial_broadcast(slots)

        #need to move channels in front
        # [(B x NUM_OF_SLOTS) x SLOT_DIM x H_init x W_init]
        x = x.permute(0,-1, 1, 2)

        # Add positional info.
        # [(B x NUM_OF_SLOTS) x SLOT_DIM x H_init x W_init]
        x = self.decoder_pos(x)

        #[(B x NUM_OF_SLOTS) x (CHANNELS + 1) x H x W]
        x = self.decoder(x)
    
        #unstack the x
        _, c, h, w = x.shape
        #[ B x NUM_OF_SLOTS x (CHANNELS + 1) x H x W]
        x = x.view(-1, self.number_of_slots, c, h, w)

        #[ B x NUM_OF_SLOTS x CHANNELS x H x W]
        #rgb image
        recons = x[:,:,:c-1,:]

        #[ B x NUM_OF_SLOTS x 1 x H x W]
        #the mask of the image
        masks = x[:,:,c-1:c,:]
        masks = self.softmax(masks)

        #resconstruc image
        #[ B x NUM_OF_SLOTS x c-1 x H x W]
        recons_image = torch.sum(recons * masks, axis=1)  # Recombine image.
        
        #output
        output = {
            "images": image,
            "generated": recons_image,
            "content": recons,
            "masks": masks,
            "loss": ((recons_image - image)**2).mean(),
            "slots": slots.cpu().detach().numpy()
        }
        return output

    def extract(self, data):
        '''
        Args:
            image: [B x C x W x H]
        Output:

        '''
        #encode the image
        #[B x SLOT_DIM x W x H]
        image = data['images']
        encoding = self.encoder(image)

        #add the positional encoding
        #[B x SLOT_DIM x W x H]
        x = self.encoder_pos(encoding)
        
        #convert to [B x (W x H) x SLOT_DIM]
        x = self.spatial_flatten(x)

        #layer norm and feed to mlp
        x = self.encoder_ln(x)

        #mapping input of [B x (W x H) x SLOT_DIM]
        #to slots of [B x (W x H) x SLOT_DIM]
        x = self.slot_mlp(x)

        # Slot Attention module.
        #to slots of [B x NUM_OF_SLOTS x SLOT_DIM]
        slots = self.slot_attention(x).cpu().detach().numpy()

        return slots

    def slot_to_img(self,slots):
        '''
        Args:
            slots: [B x NUM_OF_SLOTS x SLOT_DIM]
        Output:

        '''  

        slots = slots.to(self.device)
        
        # Spatial broadcast decoder.
        # [(B x NUM_OF_SLOTS) x H_init x W_init x SLOT_DIM]
        x = self.spatial_broadcast(slots)

        #need to move channels in front
        # [(B x NUM_OF_SLOTS) x SLOT_DIM x H_init x W_init]
        x = x.permute(0,-1, 1, 2)

        # Add positional info.
        # [(B x NUM_OF_SLOTS) x SLOT_DIM x H_init x W_init]
        x = self.decoder_pos(x)

        #[(B x NUM_OF_SLOTS) x (CHANNELS + 1) x H x W]
        x = self.decoder(x)
    
        #unstack the x
        _, c, h, w = x.shape
        #[ B x NUM_OF_SLOTS x (CHANNELS + 1) x H x W]
        x = x.view(-1, self.number_of_slots, c, h, w)

        #[ B x NUM_OF_SLOTS x CHANNELS x H x W]
        #rgb image
        recons = x[:,:,:c-1,:]

        #[ B x NUM_OF_SLOTS x 1 x H x W]
        #the mask of the image
        masks = x[:,:,c-1:c,:]
        masks = self.softmax(masks)

        #resconstruc image
        #[ B x NUM_OF_SLOTS x c-1 x H x W]
        recons_image = torch.sum(recons * masks, axis=1)  # Recombine image.
        
        #output
        output = {
            "generated": recons_image,
            "content": recons,
            "masks": masks
        }
        return output




