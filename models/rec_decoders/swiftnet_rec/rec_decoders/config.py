# TODO: Convert to json/yaml or something like that
""" Layer mapping handler """
layer_map = {
    "seg_decoder": "loaded_model.backbone.upsample", 
    "seg_spp": "loaded_model.backbone.spp", 
    "rec_decoder": "loaded_model.rec_decoder.upsample", 
    "rec_spp": "loaded_model.rec_decoder.spp_rec"
}

""" Freezing routes """
decoder_layers = {
    'lat_bw':{    
        0 : None,
        1 : ['2.bottleneck'],
        2 : ['2.bottleneck', '1.bottleneck'],
        3 : ['2.bottleneck', '1.bottleneck', '0.bottleneck'],
        4 : ['2.bottleneck', '1.bottleneck', '0.bottleneck', '2.blend_conv'],
        5 : ['2.bottleneck', '1.bottleneck', '0.bottleneck', '2.blend_conv', '1.blend_conv'],
        6 : ['0.bottleneck', '1.bottleneck', '2.bottleneck', '2.blend_conv', '1.blend_conv', '0.blend_conv'],
        7 : ['0.bottleneck', '1.bottleneck', '2.bottleneck', '2.blend_conv', '1.blend_conv', '0.blend_conv', 'spp_rec'],
        'propagation': 'Freezing propagates in lateral backward, then backward.'
    }
}