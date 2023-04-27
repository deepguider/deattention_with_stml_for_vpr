
## Pretrained checkpoints (weights) of main network
echo "1) Get checkpoint (weight) of image retrieval network with following steps:"
echo "   - Download weight_image_retrieval.tar.gz from https://drive.google.com/file/d/1xYxgii_iZogGWtKqLTQF2XlCfYguj1CY/view?usp=share_link"
echo "   - Copy the weight_image_retrieval.tar.gz to top of git dir"
echo "   - tar -zxvf weight_image_retrieval.tar.gz" 
echo "   Then you've got image_retrieval_deatt/pretrained/*"

## Pretrained weight of MobileNet which is for a segmentation.
echo "2) Get checkpoint (weight) of semantic guidance network:"
echo "   - Download weight_semantic_guidance.tar.gz from https://drive.google.com/file/d/10d0hykoqynYZZU9SDJXyKihQ0-TxhWMT/view?usp=share_link"
echo "   - Copy the weight_semantic_guidance.tar.gz to top of git dir"
echo "   - tar -zxvf weight_semantic_guidance.tar.gz" 
echo "   Then you've got image_retrieval_deatt/networks/MobileNet/pretrained/*"

cp scripts/set_gpu.bash.refer scripts/set_gpu.bash
bak_src="scripts/set_gpu.bash.bak"
bak_dst="scripts/set_gpu.bash"
if [ -e ${bak_src} ]; then
	echo "Restoring backup : cp ${bak_src} ${bak_dst}"
	cp ${bak_src} ${bak_dst}
fi
