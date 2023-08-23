# SurgicalSAM

This is the official PyTorch implementation for **[SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation], 2023**. 

Paper: https://arxiv.org/abs/2308.08746v1.

Code will be released soon!

## Abstract 
The Segment Anything Model (SAM) is a powerful foundation model that has revolutionised image segmentation. To apply SAM to surgical instrument segmentation, a common approach is to locate precise points or boxes of instruments and then use them as prompts for SAM in a zero-shot manner. However, we observe two problems with this naive pipeline: (1) the domain gap between natural objects and surgical instruments leads to poor generalisation of SAM; and (2) SAM relies on precise point or box locations for accurate segmentation, requiring either extensive manual guidance or a well-performing specialist detector for prompt preparation, which leads to a complex multi-stage pipeline. To address these problems, we introduce SurgicalSAM, a novel end-to-end efficient-tuning approach for SAM to effectively integrate surgical-specific information with SAM's pre-trained knowledge for improved generalisation. Specifically, we propose a lightweight prototype-based class prompt encoder for tuning, which directly generates prompt embeddings from class prototypes and eliminates the use of explicit prompts for improved robustness and a simpler pipeline. In addition, to address the low inter-class variance among surgical instrument categories, we propose contrastive prototype learning, further enhancing the discrimination of the class prototypes for more accurate class prompting. The results of extensive experiments on both EndoVis2018 and EndoVis2017 datasets demonstrate that SurgicalSAM achieves state-of-the-art performance while only requiring a small number of tunable parameters. 

## Datasets 
EndoVis2018 [1] and EndoVis2017 [2].

## Method
<div align="center">
  <img src="https://github.com/wenxi-yue/SurgicalSAM/assets/142413487/1102127f-d96a-4dab-8d56-541bfa213906" alt="Image 1" width="1000" height="380">
</div>

## Results 
<div align="center">
  <img src="https://github.com/wenxi-yue/SurgicalSAM/assets/142413487/0edbe1f0-3fae-4c02-b52a-d5f882c0f060" alt="Image 1" width="600" height="610">
  <img src="https://github.com/wenxi-yue/SurgicalSAM/assets/142413487/77aeb7c0-bfe5-4dc8-acce-8f4e25b7322f" alt="Image 2" width="510" height="700">
</div>

## Citation 
```
@misc{yue2023surgicalsam,
      title={SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation}, 
      author={Wenxi Yue and Jing Zhang and Kun Hu and Yong Xia and Jiebo Luo and Zhiyong Wang},
      year={2023},
      eprint={2308.08746},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## References 
[1] Allan, M.; Kondo, S.; Bodenstedt, S.; Leger, S.; Kadkhodamohammadi, R.; Luengo, I.; Fuentes, F.; Flouty, E.; Mohammed, A.; Pedersen, M.; Kori, A.; Alex, V.; Krishnamurthi, G.; Rauber, D.; Mendel,  R.; Palm, C.; Bano, S.; Saibro, G.; Shih, C.-S.; Chiang, H.-A.; Zhuang, J.; Yang, J.; Iglovikov, V.; Dobrenkii, A.; Reddiboina, M.; Reddy, A.; Liu, X.; Gao, C.; Unberath, M.; Kim, M.; Kim, C.; Kim,
C.; Kim, H.; Lee, G.; Ullah, I.; Luna, M.; Park, S. H.; Azizian, M.; Stoyanov, D.; Maier-Hein, L.; and Speidel, S. 2020. 2018 Robotic Scene Segmentation Challenge. arXiv:2001.11190.

[2] Allan, M.; Shvets, A.; Kurmann, T.; Zhang, Z.; Duggal, R.; Su, Y.-H.; Rieke, N.; Laina, I.; Kalavakonda, N.; Bodenstedt, S.; Herrera, L.; Li, W.; Iglovikov, V.; Luo, H.; Yang, J.; Stoyanov, D.; Maier-Hein, L.; Speidel, S.; and Azizian, M. 2019. 2017 Robotic Instrument Segmentation Challenge. arXiv:1902.06426.

