Aquí usamos varias redes en cascada, con **pgie** detectamos los coches, bicis, peatones y señales. Esto se le pasa a un **tracker** que sigue cada objeto para despues pasar por otra red **sgie**, esta segunda red recibe el objeto seguid por el **tracker**, es decir un crop de la imagen con solo el objeto seguido y realiza una segunda inferencia. Por ejemplo, cuando se detectan coches, se pasan por una segunda red que detecta el color, o la marca del coche

# Meta
Si se ejecuta con el flag *-m 1*, en *osd_sink_pad_buffer_probe* cuando se termina de analizar cada frame (*l_frame = batch_meta.frame_meta_list*) se empieza a analizar la información meta user list