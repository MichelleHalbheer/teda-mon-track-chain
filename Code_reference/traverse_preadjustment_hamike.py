# Gestreckter Polygonzug Ausgleichung 
# Code soll neben der Python standard Funktionen nur auf Numpy und Matplotlib Packages basieren

import numpy as np
from matplotlib import pyplot as plt

def main():
    rho=np.pi/200 # gon -> rad

    #----- Header Einstellungen -----#

    #Angenommene Genauigkeiten

    acc_dist = [0.6, 1]# [mm] und [ppm]
    acc_ang = 0.15 # [mgon]

    # Möglichkeiten Polygonzug:
    # (1) Nach Koo und Richtung angeschlossen
    #   - Aufstellen auf Start und Endpunkt -> Richtung nach Fernziel
    # (2) Nach Koo angeschlossen
    #   - Aufstellen nur auf Neu/Zwischenpunkten -> Richtung und Dist nach
    #   Start/Endpunkt messen
    # (3) Einseitung angeschlossen (fliegender PZ)
    #   - Aufstellen auf Startpunkte, Richtung nach Fernziel, kein Ende

    anschluss = 1

    #Für Analyse bei der Punktnamen gegeben sind
    custom_point_names = True
    point_label_path = 'point_labels_short_vis.txt'

    if anschluss not in [1,2,3]:
        raise ValueError("Anschluss muss 1,2,3 sein")

    # Importiere Koordinaten aus einer CSV "*.txt" Datei [Fernziel, Start, ..., Schluss, Fernziel]
    coords = np.loadtxt('koo_pz_short_vis.txt', delimiter=',') #Import
    y, x = coords[:, 0], coords[:, 1] #Aufteilen in y und x


    #----- Aufstellen A-Matrix -----#

    # [(#Aufstellungen*2) x #Koordinaten(inkl. Anschlusspunkte)]
    # Zeilen: beta1, D2, beta2, D3, beta3, ...
    # Spalten: yAnschluss,xAnschluss,yStart,xStart,y1,x1,y2,x2, ..., yEnde,xEnde, yAnschluss, xAnschluss
    A = np.zeros(((coords.shape[0] - 2) * 2, coords.size))
    
    deltas = (coords - np.insert(coords[:-1], 0, np.zeros((1,2)), axis=0))[1:] #Paarweise Differenzen berechnen (Anfangen am Fernpunkt 1)
    distances = np.diagonal(np.linalg.norm(coords[None, :, :] - coords[:, None, :], axis = -1), 1) #Distanzen zwischen allen Punkten paarweise berechnen, inkl. Fernpunkte


    ## Analytical definition of derivatives

    #d = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dd_dxi = lambda dx, dy: -dx / np.sqrt(dx**2 + dy**2) #Derivative of d with respect to xi
    dd_dxif = lambda dx, dy: dx / np.sqrt(dx**2 + dy**2) #Derivative of d with respect to xi+1
    dd_dyi = lambda dx, dy: -dy / np.sqrt(dx**2 + dy**2) #Derivative of d with respect to yi
    dd_dyif = lambda dx, dy: dy / np.sqrt(dx**2 + dy**2) #Derivative of d with respec to yi+1

    #beta = arctan((yi-1 - yi) / (xi-1 - xi)) - arctan((yi+1 - yi) / (xi+1 - xi))
    db_dxib = lambda dxb, dyb: dyb / (dxb**2 + dyb**2) #Derivative of beta with resprect to xi-1
    db_dxi = lambda dxb, dyb, dxf, dyf: - dyb / (dxb**2 + dyb**2) + dyf / (dxf**2 + dyf**2) #Derivative of beta with Respect to x
    db_dxif = lambda dxf, dyf: - dyf / (dxf**2 + dyf**2) #Derivative of beta with resprect to xi+1
    db_dyib = lambda dxb, dyb: - dxb / (dxb**2 + dyb**2) #Derivative of beta with Respect to yi-1
    db_dyi = lambda dxb, dyb, dxf, dyf: dxb / (dxb**2 + dyb**2) - dxf / (dxf**2 + dyf**2) #Derivative of beta with respect to yi
    db_dyif = lambda dxf, dyf: dxf / (dxf**2 + dyf**2) #Derivative of beta with respect to yi+1
  
    #Iteration durch A und die Ableitungen einsetzen, jeweils Richtung und Distanz
    for i in range(0,A.shape[0]//2):
        #All differences
        dyb = - deltas[i, 0]
        dxb = - deltas[i, 1]
        dyf = deltas[i+1, 0]
        dxf = deltas[i+1, 1]

        #Inset to put into A, Derivatives always have this structure
        inset = np.array([[db_dxib(dxb, dyb), db_dyib(dxb, dyb), db_dxi(dxb, dyb, dxf, dyf), db_dyi(dxb, dyb, dxf, dyf), db_dxif(dxf, dyf), db_dyif(dxf, dyf)],
                        [0, 0, dd_dxi(dxf, dyf), dd_dyi(dxf, dyf), dd_dxif(dxf, dyf), dd_dyif(dxf, dyf)]])

        #Set inset into A
        A[2*i:2*i+2, 2*i:2*i+6] = inset

    #A auf richtige Grösse ändern    
    if anschluss == 1:
        #Fernpunkt und Fixpunkt Koordinaten entfernen
        A = A[:-1, 4:-4].copy()
    elif anschluss == 2:
        #Fernpunkt und Fixpunkt Koordinaten entfernen, WInkel zu den Fernpunkten entfernen
        A = A[1:-2, 4:-4].copy()
    else:
        #Fernpunkt und Fixpunkt Koordinaten entfernen, Winkel zum letzten Fernpunkt entfernen
        A = A[:-4, 4:-4].copy()

    #Anzahl Unbekannte und Parameter
    u = A.shape[1]
    n = A.shape[0]



    #----- Aufstellen Qll- und P-Matrix -----#
    var_beta = 2 * (acc_ang / 1000 * rho)**2 #Definiere Varianz des Winkels
    var_d = lambda d: (acc_dist[0] / 1000)**2 + ((d / 1000 * acc_dist[1]) / 1000)**2 #Definiere Varianz der Distanz (mit Varianzfortpflanzung)
    #var_d = lambda d: ((acc_dist[0] / 1000) + ((d / 1000 * acc_dist[1]) / 1000))**2 #Definiere Varianz der Distanz (ohne Varianzfortpflanzung)

    #Q_ll definieren
    #NOTE: Die folgende Variante um Q_ll zu definieren ist die schnellste Variante eine alternierende Diagonalmatrix zu erstellen. Daher setze ich sie statt einer Iteration ein.
    if anschluss == 1:
        K_ll_beta = np.ones(n // 2 + 1) * var_beta #Vektor mit den Varianzen von beta
        K_ll_d = np.ones(n // 2 + 1) * var_d(distances[1:]) #Vektor mit den Varianzen von d
        K_ll = np.diag(np.vstack((K_ll_beta, K_ll_d)).ravel('F'))[:-1, :-1] #Stacken, dann Zeilenweise ravel -> Alternierende Liste, in Diagonalmatrix casten
    elif anschluss == 2:
        K_ll_beta = np.ones(n // 2 + 1) * var_beta #Vekotr mit den Varianzen von beta
        K_ll_d = np.ones(n // 2 + 1) * var_d (distances[1:-1]) #Varianzen mit den Varianzen von d
        K_ll = np.diag(np.vstack((K_ll_d, K_ll_beta)).ravel('F'))[:-1, :-1] #Stacken, dann Zeilenweise ravel -> Alternierende Liste, in Diagonalmatrix casten
    else:
        K_ll_beta = np.ones(n // 2) * var_beta #Vektor mit den Varianzen von beta
        K_ll_d = np.ones(n // 2) * var_d(distances[1:-2]) #Vektor mit den Varianzen von d
        K_ll = np.diag(np.vstack((K_ll_beta, K_ll_d)).ravel('F')) #Stacken, dann Zeilenweise ravel -> Alternierende Liste, in Diagonalmatrix casten

    sigma_0 = 1

    Q_ll = (1 / sigma_0**2) * K_ll

    P = np.linalg.inv(Q_ll)



    #----- Präanalyse durchführen -----#
    K_xx = np.linalg.inv(A.T @ P @ A)
    Q_xx = sigma_0**2 * K_xx

    ## Umrechnen in Quer- und Längsabweichung
    
    #Drehwinkel definieren
    if anschluss == 1:
        t_l = np.arctan2(y[-2] - y[1], x[-2] - x[1])
    elif anschluss == 2:
        t_l = np.arctan2(y[-3] - y[2], x[-3] - x[2])
    else:
        t_l = np.arctan2(y[-3] - y[1], x[-3] - x[1])
    F = np.array([[np.sin(t_l), -np.cos(t_l)],
                [np.cos(t_l), np.sin(t_l)]])

    #Q_xx Matrix aufteilen für jeden Punkt
    Q_list = [Q_xx[2*i:2*(i+1) , 2*i:2*(i+1)] for i in range(Q_xx.shape[0]//2)]

    #Varianz-Kovarianzmatrix der Punktkoordinaten transformieren
    transformed = [F @ Q_list[i] @ F.T for i in range(len(Q_list))]

    #Varianzen auslesen und in Liste speichern für Ausgabe und Plot
    transformed_lat = [np.sqrt(transformed[i][0,0]) for i in range(len(transformed))]
    transformed_long = [np.sqrt(transformed[i][1,1]) for i in range(len(transformed))]

    #Fixpunkt einfügen (Annahme: Koordinaten fehlerfrei bekannt)
    if anschluss in [1,3]:
        transformed_lat.insert(0, 0)
        transformed_long.insert(0, 0)
    if anschluss == 1:
        transformed_lat.append(0)
        transformed_long.append(0)
    transformed_lat = np.asarray(transformed_lat)
    transformed_long = np.asarray(transformed_long)


    #----- Ergebnisausgabe und Visualisierung-----#

    ##Vorbereitung

    #Visualisieren mit Distanzen auf der x-Achse, damit Punkte im tatsächlichen Abstand liegen
    if anschluss == 1:
        x_ticks = np.insert(np.cumsum(distances[1:-1]), 0, 0)
    elif anschluss == 2:
        x_ticks = np.insert(np.cumsum(distances[2:-2]), 0, 0)
    else:
        x_ticks = np.insert(np.cumsum(distances[1:-2]), 0, 0)

    rot_labels = 0 #Labels drehen
    align_labels = 'center' #Alignment der Labels anpassen
    if custom_point_names:
        point_names = np.loadtxt(point_label_path, dtype=str) #Eigene Punkt Labels laden falls vorhanden
        rot_labels = 45 #Drehwinkel für Labels setzen
        align_labels = 'right'
        if anschluss == 2:
            point_names = point_names[1:-1] #Punktnamen kürzen
        elif anschluss == 3:
            point_names = point_names[:-1] #Punktnamen kürzen
    else:
        point_names = np.arange(1,len(transformed_lat)+1, dtype=int) #Falls keine eigenen Labels einfach nummerieren

    ##Ergebnisse speichern
    header = 'Punktname, Sigma_x, Sigma_y, Sigma_q, Sigma_l'
    sig_x = np.asarray([np.sqrt(Q_list[i][0,0]) for i in range(len(Q_list))])
    sig_y = np.asarray([np.sqrt(Q_list[i][1,1]) for i in range(len(Q_list))])
    if anschluss in [1,3]:
        sig_x = np.insert(sig_x, 0, 0)
        sig_y = np.insert(sig_y, 0, 0)
    if anschluss == 1:
        sig_x = np.append(sig_x, 0)
        sig_y = np.append(sig_y, 0)

    output = np.stack((point_names, sig_x, sig_y, transformed_lat, transformed_long), axis=1) #Alle Arrays zusammenfügen
    np.savetxt('Genauigkeiten.csv', output, delimiter=',', header=header, fmt='%s') #In CSV-Datei abspeichern

    #Maximale Abweichung ausgeben
    print('Maximale Längsabweichung von {:.2f}mm am {}. Punkt des Polygonzugs'.format(np.max(transformed_long)*1000, np.argmax(transformed_long) + 1))
    print('Maximale Querabweichung von {:.2f}mm am {}. Punkt des Polygonzugs'.format(np.max(transformed_lat)*1000, np.argmax(transformed_lat) + 1))
    
    
    ##Plot erstellen

    #Erstellen
    plt.figure('Standardabweichung in Längs- und Querrichtung', figsize=(7,6)) #Figure erstellen
    ax = plt.subplot() #Subplot erstellen für Formatierung

    #Umwandeln der Längs und Querabweichung in Millimeter
    std_long = [std * 1000 for std in transformed_long]
    std_lat = [std * 1000 for std in transformed_lat]

    #Längs- und Querabweichung plotten
    ax.plot(x_ticks, std_long, 'b-', marker='o', fillstyle='none', label='Longitudinal')
    ax.plot(x_ticks, std_lat, 'r-', marker='o', fillstyle='none', label='Lateral')

    #Achsenbeschriftung setzen 
    ax.set_xlabel('Punktname')
    ax.set_ylabel('Standardabweichung [mm]')

    #Titel setzen, je nach Variatne
    if anschluss == 1:
        ax.set_title('Beidseitig nach Koordinaten und Orientierung angeschlossener Polygonzug', pad=20)
    elif anschluss == 2:
        ax.set_title('Beidseitig nach Koordinatne angeschlossener Polygonzug', pad=20)
    else:
        ax.set_title('Einseitig nach Koordinaten und Orientierung angeschlossener Polygonzug', pad=20)

    #Genauigkeitsangabe der Messungen
    angle = r'\alpha'
    eq = (fr'$\sigma_d={acc_dist[0]}mm + {acc_dist[1]}ppm,  \sigma_{angle} = {acc_ang}mgon$')
    plt.text(0.5, 1.02, eq, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    #x-Achse anpassen um mit Punkten übereinzustimmen
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(point_names, rotation=rot_labels, ha=align_labels, rotation_mode='anchor') #Punkte anschreiben
    plt.subplots_adjust(bottom=0.2)

    #Legende einfügen
    plt.legend(loc='best')

    ##########################################################################################
    #x und y
    plt.figure('Standardabweichung in x- und y-Richtung', figsize=(7,6)) #Figure erstellen
    ax = plt.subplot() #Subplot erstellen für Formatierung

    #Längs- und Querabweichung plotten
    ax.plot(x_ticks, sig_x, 'b-', marker='o', fillstyle='none', label='Nord-Richtung')
    ax.plot(x_ticks, sig_y, 'r-', marker='o', fillstyle='none', label='Ost-Richtung')

    #Achsenbeschriftung setzen 
    ax.set_xlabel('Punktname')
    ax.set_ylabel('Standardabweichung [mm]')

    #Titel setzen, je nach Variatne
    if anschluss == 1:
        ax.set_title('Beidseitig nach Koordinaten und Orientierung angeschlossener Polygonzug', pad=20)
    elif anschluss == 2:
        ax.set_title('Beidseitig nach Koordinatne angeschlossener Polygonzug', pad=20)
    else:
        ax.set_title('Einseitig nach Koordinaten und Orientierung angeschlossener Polygonzug', pad=20)

    #Genauigkeitsangabe der Messungen
    angle = r'\alpha'
    eq = (fr'$\sigma_d={acc_dist[0]}mm + {acc_dist[1]}ppm,  \sigma_{angle} = {acc_ang}mgon$')
    plt.text(0.5, 1.02, eq, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    #x-Achse anpassen um mit Punkten übereinzustimmen
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(point_names, rotation=rot_labels, ha=align_labels, rotation_mode='anchor')
    plt.subplots_adjust(bottom=0.2)

    #Legende einfügen
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()