# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:21:55 2022

@author: gita
"""

import gradio as gr

def image_classifier():
    # j={
    # "sentences":[
    #       {"text":"Frase ejemplo"},
    #       {"text":"Frase ejemplo"}
    # ]
    # }
    
    # j = {
    #     'text':"Frase ejemplo Frase ejemplo ", 
        
    #     'text_labeled':" \"Frase\"/Entity_Type ejemplo \"Frase\"/Entity_Type ejemplo ",
        
    #     'sentences':[
    #         {'text':"Frase ejemplo", 
    #           'text_labeled':" \"Frase\"/Entity_Type ejemplo", 
    #           'tokens':[
    #               {'text':"Frase", 'label':"Entity_Type"},
    #               {'text':"ejemplo", 'label':"O"}
    #               ]},
            
    #         {'text':"Frase ejemplo", 
    #           'text_labeled':" \"Frase\"/Entity_Type ejemplo", 
    #           'tokens':[
    #               {'text':"Frase", 'label':"Entity_Type"},
    #               {'text':"ejemplo", 'label':"O"}
    #               ]}
            
    #         ], 
        
        
    #     'entities': [
    #         {
    #             'entity': "Entity_Type" ,
    #             'index' : 0,
    #             'word' : "Frase",
    #             'start': 0,
    #             'end' : 5
                
    #             },
    #         {
    #             'entity': "Entity_Type" ,
    #             'index' : 2,
    #             'word' : "Frase",
    #             'start': 14,
    #             'end' : 19
                
    #             }
    #         ]
        
    #     }
    
    
    j = {
    
        'text':"Frase ejemplo Frase ejemplo", 

        'sentences':[
            {'text':"Frase ejemplo", 
              'id':"s0", 
              'tokens':[
                  {'text':"Frase", 'begin':0, 'end':5},
                  {'text':"ejemplo", 'begin':6, 'end':13}
                  ]},
            
            {'text':"Frase ejemplo", 
              'id':"s1", 
              'tokens':[
                  {'text':"Frase", 'begin':14, 'end':19},
                  {'text':"ejemplo", 'begin':20, 'end':27}
                  ]},
            
            ], 
        
        
        'mentions': [
            {
                'id': "s0-m0" ,
                'type' : "Entity_type",
                'begin' : 0,
                'end': 5,
                
                },
    
            {
                'id': "s1-m0" ,
                'type' : "Entity_type",
                'begin' : 14,
                'end': 19,
                
                }
    
            ]
        
        }
    
    
    
    return j

demo = gr.Interface(fn=image_classifier, inputs=None, outputs=gr.JSON())
demo.launch()

#%% 
# JSON FORMAT OUTPUT

# Document:{ text:"Texto"
          
#           text_labeled: "Texto \ENTITY"
          
#           sentences:[{ text:"Texto"
          
    #           text_labeled: "Texto \ENTITY"
              
    #           tokens: [ {text:"Texto", label : "ENTITY"},
    #                    {text:"Texto", label : "ENTITY"},
    #                    {text:"Texto", label : "ENTITY"}
                  
    #               ] 
                  
    #               },
      
    #            { text:"Texto"
              
    #           text_labeled: "Texto <ENTITY>"
              
    #           tokens: [ {text:"Texto", label : "ENTITY"},
    #                    {text:"Texto", label : "ENTITY"},
    #                    {text:"Texto", label : "ENTITY"}
              
#               ]
 
#               }          
#            ],
            # entities:[
            #     {
            #         'entity': "ENTITY",
            #         'index': num,
            #         'word': "Texto",
            #         'start': num,
            #         'end' : num
            #     }
            #     ]
#     }

#%% 

# JSON FORMAT INPUT

# json{...
#       sentences:{
#           s:{
#               text:
#               }
#                 }
     
#       ...}