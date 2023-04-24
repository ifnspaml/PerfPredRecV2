layer_map = {
    "seg_decoder": "loaded_model.backbone.decoder", 
    "seg_spp"    : "loaded_model.backbone.aspp", 
    "rec_decoder": "loaded_model.rec_decoder", 
    "rec_spp"    : "loaded_model.aspp"
}

""" Freezing routes """
decoder_layers = {
    'bw':{    
        0: None,
        1: ['last_conv.1'],
        2: ['last_conv.1', 'last_conv.0'],
        3: ['last_conv.1', 'last_conv.0', 'aspp'],
        4: ['last_conv.1', 'last_conv.0', 'aspp', 'conv1', 'bn1'],
        'propagation': 'Freezing propagates backwards.'
    },
}