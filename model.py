import segmentation_models_pytorch as smp 


def get_model():
    model = smp.Unet(
        encoder_name="resnet34",          
        encoder_weights="imagenet",       
        in_channels=1,                    
        classes=1,                        
        activation=None                   
    )
    return model


'''
def get_model():
    model = smp.DeepLabV3(
        encoder_name="resnet34",   
        encoder_weights="imagenet",       
        in_channels=1,                    
        classes=1,                        
        activation=None                   
    )
    return model


'''
'''
def get_model():
    model = smp.Linknet(
        encoder_name="resnet34",      
        encoder_weights="imagenet",   
        in_channels=1,                
        classes=1,                    
        activation=None               
    )
    return model
'''

