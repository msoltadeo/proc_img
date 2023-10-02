import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

def local_heq(img, m, n, border):
    '''Equalizar una imagen localmente
    img: imagen a equalizar
    m: numero de filas del filtro
    n: numero de columnas del filtro
    border: tipo de border a aplicar en el padding'''
    img_pad = cv2.copyMakeBorder(img,  int((m - 1)/2), int((m - 1)/2), int((n - 1)/2), int((n - 1)/2), border)
    M, N = img_pad.shape
    step_x = int((m-1)/2)
    step_y = int((n-1)/2)
    local_heq = np.zeros([M, N])
    for i in range(M-step_x):
        for j in range(N-step_y):
            local = img_pad[i:i+m, j:j+n]
            img_heq = cv2.equalizeHist(local)
            local_heq[i + step_x, j + step_y] = img_heq[step_x, step_y]
    return local_heq[int((m - 1)/2):-int((m - 1)/2), int((n - 1)/2): -int((n - 1)/2)]

def test_local_equalization(image):
    m_n = np.array([[(3,3), (9,9), (21,21)], [(33,33), (43,43), (53,53)]])
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #que era cv2.IMREAD_GRAYSCALE
    border = cv2.BORDER_REPLICATE

    fig_num = 1
    for m, n in m_n.reshape(1, m_n.shape[0]*m_n.shape[1], 2)[0]:
        img_local_heq = local_heq(img, m, n, border)
        subplot_pos = int('{}{}{}'.format(m_n.shape[0], m_n.shape[1], fig_num))
        if fig_num == 1:
            ax1 = plt.subplot(subplot_pos)
        else:
            plt.subplot(subplot_pos, sharex=ax1, sharey=ax1)
        plt.imshow(img_local_heq, cmap='gray')
        plt.title('filter dimensions {}x{}'.format(m,n))
        fig_num += 1
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type = str, help = 'path for the image')
    args = parser.parse_args()

    image = args.image_path
    test_local_equalization(image)
