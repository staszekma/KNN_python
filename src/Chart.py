import matplotlib.pyplot as plt

def Chart2D(x1:int, y1, x2=[], y2=[], x3=[], y3=[], x4=[], y4=[]): 
    
    c_1, c_2, c_3, c_4 = ["red"], ["green"], ["blue"], ["yellow"]
    l_1, l_2, l_3, l_4 = len(x1), len(x2), len(x3), len(x4)

    colors = c_1*l_1+c_2*l_2+c_3*l_3+c_4*l_4
    x, y = x1+x2+x3+x4, y1+y2+y3+y4
    
    fig, ax = plt.subplots()
    
    ax.scatter(x, y, c=colors)
    plt.show()


def Chart2D_c(x, y, color=[0,0,0,0]):
    Acc = 0
    l_x=len(x)
    for c in color:
        Acc = Acc+c

    if Acc!=l_x:
        return 1

    
    c_1, c_2, c_3, c_4 = ["red"], ["green"], ["blue"], ["yellow"]

    colors = c_1*color[1]+c_2*color[2]+c_3*color[3]+c_4*color[3]
    
    fig, ax = plt.subplots()
    
    ax.scatter(x, y, c=colors)
    plt.show()


def Chart3D(x1, y1, z1, x2=[], y2=[], z2=[], x3=[], y3=[], z3=[], x4=[], y4=[], z4=[]):
    c_1, c_2, c_3, c_4 = ["red"], ["green"], ["blue"], ["yellow"]
    l_1, l_2, l_3, l_4 = len(x1), len(x2), len(x3), len(x4)

    colors = c_1*l_1+c_2*l_2+c_3*l_3+c_4*l_4
    x, y, z = x1+x2+x3+x4, y1+y2+y3+y4, z1+z2+z3+z4
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z, c=colors)
    plt.show()


def main():
    Chart2D_c([2,1],[1,2])


if __name__ == "__main__":
    main()