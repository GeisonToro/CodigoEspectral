# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import math
import numpy as np
import re
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import numexpr as ne
#from ttkthemes import ThemedTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# -------------------------------------------------------------------------
# COMPILAR UNA ÚNICA VEZ EL PATRÓN DE TOKENS PARA GANAR RENDIMIENTO
# -------------------------------------------------------------------------
_PAT_TOKEN = re.compile(r"[0-9]*\.?[0-9]+|[a-d]|[()\+\-\*/\^]|[^\s]")

# -------------------------------------------------------------------------
# FUNCIONES DE ÍNDICES DE VEGETACIÓN (SIN CAMBIOS)
# -------------------------------------------------------------------------

def calcular_savi(nir, rojo, L=0.5):
    return (1 + L) * (((nir)/100) - ((rojo)/100)) / (((nir)/100) + ((rojo)/100) + L)

def calcular_ndvi(nir, rojo):
    return (nir - rojo) / (nir + rojo)

def calcular_evi(nir, rojo, azul, L=1, C1=6, C2=7.5, G=2.5):
    return G * (((nir)/100) - ((rojo)/100)) / (((nir)/100) + C1*((rojo)/100) - C2*((azul)/100) + L)

def calcular_nbr(nir, swir2):
    return (nir - swir2) / (nir + swir2)

def calcular_gli(verde, rojo, azul):
    return (2*((verde)/100) - ((rojo)/100) - ((azul)/100)) / (2*((verde)/100) + ((rojo)/100) + ((azul)/100))

def calcular_gcl(nir, verde):
    return (((nir)/100) / ((verde)/100)) - 1

def calcular_sipi(nir, azul, rojo):
    return (((nir)/100) - ((azul)/100)) / (((nir)/100) - ((rojo)/100))

def calcular_mcari(rojo, verde, nir):
    return ((((nir)/100) - ((rojo)/100)) - 0.2*(((nir)/100) - ((verde)/100))) * (((nir)/100) / ((rojo)/100))

# -------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------------------

def obtener_reflectancia_en_onda(longitudes, reflectancias, onda_objetivo, tolerancia=2.0):
    if len(longitudes) == 0:
        return None
    idx = np.argmin(np.abs(longitudes - onda_objetivo))
    return reflectancias[idx] if abs(longitudes[idx] - onda_objetivo) <= tolerancia else None


def calcular_indice_personalizado(formula: str, constantes: dict, longitudes: np.ndarray, reflectancias: np.ndarray):
    """Evalúa un índice de vegetación personalizado.

    Se vectoriza la búsqueda de reflectancias para los tokens numéricos
    para evitar bucles Python innecesarios.
    """
    # Normalizar exponentes
    expr = formula.replace("^", "**")

    # Tokenizar con el patrón pre‑compilado (ahorro ≈25% de tiempo en benchmarks)
    tokens = _PAT_TOKEN.findall(expr)

    # Identificar tokens numéricos en bloque (evita múltiples float() en try/except)
    mascaras_num = np.fromiter((t.replace('.', '', 1).isdigit() for t in tokens), bool)
    if mascaras_num.any():
        # Convertir todos los tokens numéricos de una vez
        valores = np.array(tokens, dtype=object)
        nums = valores[mascaras_num].astype(float)
        # Para cada longitud solicitada buscamos el índice más cercano (vectorizado)
        idxs = np.abs(nums[:, None] - longitudes).argmin(axis=1)
        refls = reflectancias[idxs].astype(str)
        valores[mascaras_num] = refls
        tokens = valores.tolist()

    # Sustituir constantes a,b,c,d
    tokens = [str(constantes.get(t, t)) for t in tokens]

    expresion_final = "".join(tokens)
    try:
        res = ne.evaluate(expresion_final)
        return float(res) if isinstance(res, (int, float, np.floating)) else None
    except ne.NumExprError:
        return None
    except Exception as e:
        messagebox.showerror("Error", f"Expresión personalizada inválida: {e}")
        return None


# -------------------------------------------------------------------------
# VENTANA PARA ÍNDICE PERSONALIZADO (VentanaIndicePersonalizado)
# -------------------------------------------------------------------------
class VentanaIndicePersonalizado(tk.Toplevel):
    def __init__(self, padre, al_añadir_callback):
        super().__init__(padre)
        self.title("Añadir Índice de Vegetación")
        self.al_añadir_callback = al_añadir_callback

        marco_principal = tk.Frame(self)
        marco_principal.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        tk.Label(marco_principal, text="Nombre de Índice:").pack(anchor="w")
        self.entrada_nombre = tk.Entry(marco_principal, width=30)
        self.entrada_nombre.pack(padx=5, pady=5)

        tk.Label(marco_principal, text="Fórmula:").pack(anchor="w")
        self.entrada_formula = tk.Entry(marco_principal, width=40)
        self.entrada_formula.pack(padx=5, pady=5)

        marco_calculadora = tk.Frame(marco_principal)
        marco_calculadora.pack(padx=5, pady=5, fill=tk.BOTH)

        fila1 = tk.Frame(marco_calculadora); fila1.pack(anchor="w", pady=2)
        for texto in ["7","8","9","("]:
            tk.Button(fila1, text=texto, width=5, height=2,
                      command=lambda t=texto: self.insertar_formula(t)).pack(side=tk.LEFT, padx=3)

        fila2 = tk.Frame(marco_calculadora); fila2.pack(anchor="w", pady=2)
        for texto in ["4","5","6",")"]:
            tk.Button(fila2, text=texto, width=5, height=2,
                      command=lambda t=texto: self.insertar_formula(t)).pack(side=tk.LEFT, padx=3)

        fila3 = tk.Frame(marco_calculadora); fila3.pack(anchor="w", pady=2)
        for texto in ["1","2","3","+"]:
            tk.Button(fila3, text=texto, width=5, height=2,
                      command=lambda t=texto: self.insertar_formula(t)).pack(side=tk.LEFT, padx=3)

        fila4 = tk.Frame(marco_calculadora); fila4.pack(anchor="w", pady=2)
        for texto in ["0",".","^","-"]:
            tk.Button(fila4, text=texto, width=5, height=2,
                      command=lambda t=texto: self.insertar_formula(t)).pack(side=tk.LEFT, padx=3)

        fila5 = tk.Frame(marco_calculadora); fila5.pack(anchor="w", pady=2)
        for texto in ["*","/"]:
            tk.Button(fila5, text=texto, width=5, height=2,
                      command=lambda t=texto: self.insertar_formula(t)).pack(side=tk.LEFT, padx=3)

        marco_constantes = tk.Frame(marco_principal, bd=2, relief=tk.GROOVE)
        marco_constantes.pack(padx=5, pady=5, fill=tk.X)
        tk.Label(marco_constantes, text="Constantes (a,b,c,d):").pack(anchor="w", padx=5, pady=(5,0))

        self.entradas_const = {}
        filaC = tk.Frame(marco_constantes)
        filaC.pack(anchor="w", pady=5)
        for cst in ["a","b","c","d"]:
            sub = tk.Frame(filaC)
            sub.pack(side=tk.LEFT, padx=5)
            tk.Label(sub, text=f"{cst} =").pack(side=tk.LEFT)
            entrada = tk.Entry(sub, width=4)
            entrada.pack(side=tk.LEFT, padx=2)
            self.entradas_const[cst] = entrada

        tk.Button(marco_principal, text="Añadir", width=10,
                  command=self.al_añadir_click).pack(pady=10)

    def insertar_formula(self, texto):
        actual = self.entrada_formula.get()
        self.entrada_formula.delete(0, tk.END)
        self.entrada_formula.insert(0, actual + texto)

    def al_añadir_click(self):
        nombre = self.entrada_nombre.get().strip()
        formula = self.entrada_formula.get().strip()
        if not nombre:
            messagebox.showwarning("Advertencia", "Por favor, ingresa un nombre para el índice.")
            return
        if not formula:
            messagebox.showwarning("Advertencia", "Por favor, ingresa una fórmula para el índice.")
            return

        dicc_const = {}
        for cst in ["a","b","c","d"]:
            valor_str = self.entradas_const[cst].get().strip()
            if valor_str:
                try:
                    dicc_const[cst] = float(valor_str)
                except:
                    dicc_const[cst] = None

        self.al_añadir_callback(nombre, formula, dicc_const)
        self.destroy()

# -------------------------------------------------------------------------
# CLASE VENTANA PRINCIPAL (VentanaPrincipal)
# -------------------------------------------------------------------------
class VentanaPrincipal:
    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title("Aplicación Espectral")

        self.archivos_datos = {}       # Diccionario: { ruta_archivo: {...} }
        self.archivos_en_grafico = []  # Lista de rutas "añadidas al gráfico"

        contenedor_principal = tk.Frame(self.raiz)
        contenedor_principal.pack(fill=tk.BOTH, expand=True)

        self.canvas_principal = tk.Canvas(contenedor_principal)
        self.canvas_principal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar_y = tk.Scrollbar(contenedor_principal, orient=tk.VERTICAL,
                                   command=self.canvas_principal.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_principal.configure(yscrollcommand=scrollbar_y.set)

        self.marco_interno = tk.Frame(self.canvas_principal)
        self.canvas_principal.create_window((0,0), window=self.marco_interno, anchor="nw")

        self.marco_interno.bind("<Configure>", lambda e:
            self.canvas_principal.configure(scrollregion=self.canvas_principal.bbox("all")))

        marco_superior = tk.Frame(self.marco_interno)
        marco_superior.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5, expand=True)

        marco_inferior = tk.Frame(self.marco_interno)
        marco_inferior.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ------------------- Área Izquierda -------------------
        area_izquierda = tk.Frame(marco_superior)
        area_izquierda.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Button(area_izquierda, text="Cargar Archivos a Lista",
                  command=self.cargar_archivos).pack(pady=5)
        tk.Label(area_izquierda, text="Archivos Cargados", font=("Arial",10,"bold")).pack(pady=5)

        marco_archivos = tk.Frame(area_izquierda, width=300, height=200)
        marco_archivos.pack_propagate(False)
        marco_archivos.pack(fill=tk.X, pady=5)

        xscroll_archivos = tk.Scrollbar(marco_archivos, orient=tk.HORIZONTAL)
        xscroll_archivos.pack(side=tk.BOTTOM, fill=tk.X)

        self.lista_archivos = tk.Listbox(marco_archivos, selectmode=tk.EXTENDED,
                                         xscrollcommand=xscroll_archivos.set, height=10)
        self.lista_archivos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_archivos.config(command=self.lista_archivos.xview)

        tk.Button(area_izquierda, text="Eliminar Archivos",
                  command=self.eliminar_archivos).pack(pady=5)

        linea_btn1 = tk.Frame(area_izquierda)
        linea_btn1.pack(pady=5)
        tk.Button(linea_btn1, text="Añadir al Gráfico",
                  command=self.anadir_al_grafico).pack(side=tk.LEFT, padx=5)
        tk.Button(linea_btn1, text="Quitar del Gráfico",
                  command=self.quitar_del_grafico).pack(side=tk.LEFT, padx=5)

        linea_btn2 = tk.Frame(area_izquierda)
        linea_btn2.pack(pady=5)
        tk.Button(linea_btn2, text="Añadir Todos al Gráfico",
                  command=self.anadir_todos_al_grafico).pack(side=tk.LEFT, padx=5)
        tk.Button(linea_btn2, text="Quitar Todos del Gráfico",
                  command=self.quitar_todos_del_grafico).pack(side=tk.LEFT, padx=5)

        tk.Label(area_izquierda, text="").pack(pady=10)
        tk.Button(area_izquierda, text="Procesar Datos",
                  command=self.abrir_ventana_procesamiento).pack(pady=10)

        # ------------------- Área Derecha -------------------
        area_derecha = tk.Frame(marco_superior)
        area_derecha.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tabla de Lecturas
        marco_tabla_lecturas = tk.Frame(area_derecha, width=600, height=200)
        marco_tabla_lecturas.pack_propagate(False)
        marco_tabla_lecturas.pack(fill=tk.X, padx=5, pady=5)

        xscroll_lecturas = ttk.Scrollbar(marco_tabla_lecturas, orient=tk.HORIZONTAL)
        xscroll_lecturas.pack(side=tk.BOTTOM, fill=tk.X)
        yscroll_lecturas = ttk.Scrollbar(marco_tabla_lecturas, orient=tk.VERTICAL)
        yscroll_lecturas.pack(side=tk.RIGHT, fill=tk.Y)

        self.tabla_lecturas = ttk.Treeview(marco_tabla_lecturas, show="headings",
                                           xscrollcommand=xscroll_lecturas.set,
                                           yscrollcommand=yscroll_lecturas.set, height=8)
        self.tabla_lecturas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_lecturas.config(command=self.tabla_lecturas.xview)
        yscroll_lecturas.config(command=self.tabla_lecturas.yview)

        tk.Button(area_derecha, text="Exportar Lecturas",
                  command=self.exportar_lecturas).pack(pady=5)

        # Tabla de Estadísticas
        marco_tabla_estadisticas = tk.Frame(area_derecha, width=600, height=160)
        marco_tabla_estadisticas.pack_propagate(False)
        marco_tabla_estadisticas.pack(fill=tk.X, padx=5, pady=5)

        xscroll_estadisticas = ttk.Scrollbar(marco_tabla_estadisticas, orient=tk.HORIZONTAL)
        xscroll_estadisticas.pack(side=tk.BOTTOM, fill=tk.X)
        self.tabla_estadisticas = ttk.Treeview(marco_tabla_estadisticas, show="headings",
                                               xscrollcommand=xscroll_estadisticas.set, height=7)
        self.tabla_estadisticas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_estadisticas.config(command=self.tabla_estadisticas.xview)

        tk.Button(area_derecha, text="Exportar Estadísticas",
                  command=self.exportar_estadisticas).pack(pady=5)

        # Gráfico principal
        self.figura_principal = Figure(figsize=(8,4))
        self.eje_principal = self.figura_principal.add_subplot(111)
        self.eje_principal.set_xlabel("Longitud de Onda")
        self.eje_principal.set_ylabel("Reflectancia")

        self.lienzo_principal = FigureCanvasTkAgg(self.figura_principal, master=marco_inferior)
        self.lienzo_principal.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.barra_herramientas = NavigationToolbar2Tk(self.lienzo_principal, marco_inferior)
        self.barra_herramientas.update()
        self.barra_herramientas.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.anotacion = self.eje_principal.annotate("", xy=(0,0), xytext=(20,20),
                                                     textcoords="offset points",
                                                     bbox=dict(boxstyle="round", fc="w"),
                                                     arrowprops=dict(arrowstyle="->"))
        self.anotacion.set_visible(False)
        self.lienzo_principal.mpl_connect("motion_notify_event", self.mover_cursor)

        self.inicializar_tablas()

    # -------------------------------------------------------
    # Inicializar tablas de lecturas y estadísticas
    # -------------------------------------------------------
    def inicializar_tablas(self):
        self.tabla_lecturas["columns"] = ("Longitud de Onda",)
        self.tabla_lecturas.heading("Longitud de Onda", text="Longitud de Onda")
        self.tabla_lecturas.column("Longitud de Onda", width=120)

        self.tabla_estadisticas["columns"] = ("Estadística",)
        self.tabla_estadisticas.heading("Estadística", text="Estadística")
        self.tabla_estadisticas.column("Estadística", width=200)

        etiquetas_estadisticas = [
            "Reflectancia Mínima",
            "Longitud de Onda de Reflectancia Mínima",
            "Promedio",
            "Reflectancia Máxima",
            "Longitud de Onda de Reflectancia Máxima",
            "Desviación Estándar"
        ]
        for lbl in etiquetas_estadisticas:
            self.tabla_estadisticas.insert("", tk.END, values=(lbl,))

    # -------------------------------------------------------
    # Carga y eliminación de archivos
    # -------------------------------------------------------
    def cargar_archivos(self):
        rutas = filedialog.askopenfilenames(filetypes=[("Archivos TRM","*.trm"), ("Todos","*.*")])
        if not rutas:
            return
        for ruta in rutas:
            nombre_base = os.path.basename(ruta)
            if ruta not in self.archivos_datos:
                lista_ondas, lista_refl = [], []
                try:
                    with open(ruta, 'r') as archivo:
                        for linea in archivo:
                            linea = linea.strip()
                            if not linea:
                                continue
                            partes = linea.split()
                            if len(partes) == 2:
                                try:
                                    w = float(partes[0])
                                    r = float(partes[1])
                                    lista_ondas.append(w)
                                    lista_refl.append(r)
                                except:
                                    pass
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo leer el archivo {nombre_base}:\n{e}")
                    continue

                if len(lista_ondas)>0:
                    arr_ondas = np.array(lista_ondas, dtype=float)
                    arr_ref   = np.array(lista_refl,  dtype=float)
                    self.archivos_datos[ruta] = {
                        'longitudes': arr_ondas,
                        'reflectancias': arr_ref,
                        'nombre_mostrar': nombre_base
                    }
                    self.lista_archivos.insert(tk.END, nombre_base)

    def eliminar_archivos(self):
        seleccion = self.lista_archivos.curselection()
        if not seleccion:
            return
        nombres_eliminar = [self.lista_archivos.get(i) for i in seleccion]
        claves_eliminar  = []

        for k,v in self.archivos_datos.items():
            if v['nombre_mostrar'] in nombres_eliminar:
                claves_eliminar.append(k)

        for c in claves_eliminar:
            del self.archivos_datos[c]

        for ne in nombres_eliminar:
            idx = None
            for i in range(self.lista_archivos.size()):
                if self.lista_archivos.get(i) == ne:
                    idx = i
                    break
            if idx is not None:
                self.lista_archivos.delete(idx)

        quitados_grafico = [f for f in self.archivos_en_grafico if f not in self.archivos_datos]
        for q in quitados_grafico:
            self.archivos_en_grafico.remove(q)

        self.actualizar_tabla_lecturas()
        self.actualizar_tabla_estadisticas()
        self.actualizar_grafico_principal()

    # -------------------------------------------------------
    # Añadir/Quitar archivos al gráfico
    # -------------------------------------------------------
    def anadir_al_grafico(self):
        seleccion = self.lista_archivos.curselection()
        if not seleccion:
            return
        nombres_sel = [self.lista_archivos.get(i) for i in seleccion]
        claves_sel  = [k for k,v in self.archivos_datos.items() if v['nombre_mostrar'] in nombres_sel]
        for ruta in claves_sel:
            if ruta not in self.archivos_en_grafico:
                self.archivos_en_grafico.append(ruta)
        self.actualizar_tabla_lecturas()
        self.actualizar_tabla_estadisticas()
        self.actualizar_grafico_principal()

    def quitar_del_grafico(self):
        seleccion = self.lista_archivos.curselection()
        if not seleccion:
            return
        nombres_sel = [self.lista_archivos.get(i) for i in seleccion]
        claves_sel  = [k for k,v in self.archivos_datos.items() if v['nombre_mostrar'] in nombres_sel]
        for ruta in claves_sel:
            if ruta in self.archivos_en_grafico:
                self.archivos_en_grafico.remove(ruta)
        self.actualizar_tabla_lecturas()
        self.actualizar_tabla_estadisticas()
        self.actualizar_grafico_principal()

    def anadir_todos_al_grafico(self):
        for ruta in self.archivos_datos.keys():
            if ruta not in self.archivos_en_grafico:
                self.archivos_en_grafico.append(ruta)
        self.actualizar_tabla_lecturas()
        self.actualizar_tabla_estadisticas()
        self.actualizar_grafico_principal()

    def quitar_todos_del_grafico(self):
        self.archivos_en_grafico.clear()
        self.actualizar_tabla_lecturas()
        self.actualizar_tabla_estadisticas()
        self.actualizar_grafico_principal()

    # -------------------------------------------------------
    # Actualizar tablas y gráfico principal
    # -------------------------------------------------------
    def actualizar_tabla_lecturas(self):
        self.tabla_lecturas.delete(*self.tabla_lecturas.get_children())
        if len(self.archivos_en_grafico) == 0:
            self.tabla_lecturas["columns"] = ("Longitud de Onda",)
            self.tabla_lecturas.heading("Longitud de Onda", text="Longitud de Onda")
            self.tabla_lecturas.column("Longitud de Onda", width=120)
            # Quitar pseudo-archivo "Promedio"
            for k,v in list(self.archivos_datos.items()):
                if v['nombre_mostrar'] == "Promedio":
                    del self.archivos_datos[k]
            return

        nombres_base = [self.archivos_datos[f]['nombre_mostrar'] for f in self.archivos_en_grafico]
        columnas     = ["Longitud de Onda"] + nombres_base + ["promedio"]
        self.tabla_lecturas["columns"] = columnas
        for col in columnas:
            self.tabla_lecturas.heading(col, text=col)
            self.tabla_lecturas.column(col, width=120)

        primer_archivo = self.archivos_en_grafico[0]
        ondas_base     = self.archivos_datos[primer_archivo]['longitudes']
        lista_refl     = [self.archivos_datos[f]['reflectancias'] for f in self.archivos_en_grafico]

        lista_promedios = []
        for i, onda in enumerate(ondas_base):
            vals = [reflect[i] for reflect in lista_refl]
            prom = sum(vals)/len(vals)
            valores_fila = [onda] + vals + [prom]
            lista_promedios.append(prom)
            self.tabla_lecturas.insert("", tk.END, values=tuple(valores_fila))

        pseudo_clave = "__promedio__"
        if pseudo_clave not in self.archivos_datos:
            self.archivos_datos[pseudo_clave] = {
                'longitudes': ondas_base.copy(),
                'reflectancias': np.array(lista_promedios, dtype=float),
                'nombre_mostrar': "Promedio"
            }
        else:
            self.archivos_datos[pseudo_clave]['longitudes'] = ondas_base.copy()
            self.archivos_datos[pseudo_clave]['reflectancias'] = np.array(lista_promedios, dtype=float)

    def actualizar_tabla_estadisticas(self):
        self.tabla_estadisticas.delete(*self.tabla_estadisticas.get_children())
        etiquetas_estadisticas = [
            "Reflectancia Mínima",
            "Longitud de Onda de Reflectancia Mínima",
            "Promedio",
            "Reflectancia Máxima",
            "Longitud de Onda de Reflectancia Máxima",
            "Desviación Estándar"
        ]
        if len(self.archivos_en_grafico) == 0:
            self.tabla_estadisticas["columns"] = ("Estadística",)
            self.tabla_estadisticas.heading("Estadística", text="Estadística")
            self.tabla_estadisticas.column("Estadística", width=200)
            for lbl in etiquetas_estadisticas:
                self.tabla_estadisticas.insert("", tk.END, values=(lbl,))
            return

        nombres_base= [self.archivos_datos[f]['nombre_mostrar'] for f in self.archivos_en_grafico]
        columnas    = ["Estadística"] + nombres_base + ["promedio"]
        self.tabla_estadisticas["columns"] = columnas
        for col in columnas:
            self.tabla_estadisticas.heading(col, text=col)
            w = 200 if col == "Estadística" else 120
            self.tabla_estadisticas.column(col, width=w)

        primer_archivo = self.archivos_en_grafico[0]
        ondas          = self.archivos_datos[primer_archivo]['longitudes']
        todos_refl     = [self.archivos_datos[f]['reflectancias'] for f in self.archivos_en_grafico]

        proms_col = [sum(r[i] for r in todos_refl)/len(todos_refl)
                     for i in range(len(ondas))]

        def calc_stats(refs, wls):
            r_min = min(refs)
            idx_min = refs.tolist().index(r_min)
            wl_min = wls[idx_min]

            r_max = max(refs)
            idx_max = refs.tolist().index(r_max)
            wl_max = wls[idx_max]

            r_mean = np.mean(refs)
            var    = np.mean((refs - r_mean)**2)
            std_dev= math.sqrt(var)
            return (r_min, wl_min, r_mean, r_max, wl_max, std_dev)

        stats_archivos = [calc_stats(r, ondas) for r in todos_refl]
        stats_prom     = calc_stats(np.array(proms_col), ondas)

        for i, lbl in enumerate(etiquetas_estadisticas):
            fila = [lbl]
            for st in stats_archivos:
                fila.append(f"{st[i]:.4f}")
            fila.append(f"{stats_prom[i]:.4f}")
            self.tabla_estadisticas.insert("", tk.END, values=tuple(fila))

    def actualizar_grafico_principal(self):
        self.eje_principal.clear()
        self.eje_principal.set_xlabel("Longitud de Onda")
        self.eje_principal.set_ylabel("Reflectancia")

        if len(self.archivos_en_grafico) == 0:
            self.lienzo_principal.draw()
            return

        colores = ["red","blue","green","orange","purple","brown","cyan","magenta"]
        for i, ruta in enumerate(self.archivos_en_grafico):
            ondas = self.archivos_datos[ruta]['longitudes']
            refls = self.archivos_datos[ruta]['reflectancias']
            nom   = self.archivos_datos[ruta]['nombre_mostrar']
            c     = colores[i % len(colores)]
            self.eje_principal.plot(ondas, refls, color=c, label=nom)

            r_min = min(refls)
            idx_mi= refls.tolist().index(r_min)
            wl_min= ondas[idx_mi]

            r_max = max(refls)
            idx_ma= refls.tolist().index(r_max)
            wl_max= ondas[idx_ma]

            self.eje_principal.plot(wl_min, r_min, marker="o", color=c)
            self.eje_principal.plot(wl_max, r_max, marker="o", color=c)

        self.eje_principal.legend()
        self.lienzo_principal.draw()

    def mover_cursor(self, evento):
        # Futura mejoras
        pass

    # -------------------------------------------------------
    # Exportar Lecturas y Estadísticas
    # -------------------------------------------------------
    def exportar_lecturas(self):
        if len(self.tabla_lecturas["columns"]) <= 1:
            messagebox.showinfo("Información", "No hay datos para exportar.")
            return
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".txt",
                                                     initialfile="Lecturas.txt",
                                                     filetypes=[("Archivos de texto", "*.txt")])
        if not ruta_guardado:
            return
        columnas = self.tabla_lecturas["columns"]
        with open(ruta_guardado, 'w') as f:
            f.write("\t".join(columnas) + "\n")
            for fila_id in self.tabla_lecturas.get_children():
                valores = self.tabla_lecturas.item(fila_id, "values")
                fila_str= "\t".join(str(x) for x in valores)
                f.write(fila_str+"\n")
        messagebox.showinfo("Éxito", "Lecturas exportadas correctamente.")

    def exportar_estadisticas(self):
        if len(self.tabla_estadisticas["columns"]) <= 1:
            messagebox.showinfo("Información", "No hay estadísticas para exportar.")
            return
        ruta_guardado = filedialog.asksaveasfilename(defaultextension=".txt",
                                                     initialfile="Estadisticas.txt",
                                                     filetypes=[("Archivos de texto", "*.txt")])
        if not ruta_guardado:
            return
        columnas = self.tabla_estadisticas["columns"]
        with open(ruta_guardado, 'w') as f:
            f.write("\t".join(columnas) + "\n")
            for fila_id in self.tabla_estadisticas.get_children():
                valores = self.tabla_estadisticas.item(fila_id, "values")
                fila_str= "\t".join(str(x) for x in valores)
                f.write(fila_str+"\n")
        messagebox.showinfo("Éxito", "Estadísticas exportadas correctamente.")

    # -------------------------------------------------------
    # Abrir la Ventana de Procesamiento (segunda)
    # -------------------------------------------------------
    def abrir_ventana_procesamiento(self):
        vp = VentanaProcesamiento(self)
        vp.geometry("1050x700")
        vp.mainloop()

# -------------------------------------------------------------------------
# VENTANA DE PROCESAMIENTO (VentanaProcesamiento)
# -------------------------------------------------------------------------
class VentanaProcesamiento(tk.Toplevel):
    def __init__(self, ventana_principal):
        super().__init__(ventana_principal.raiz)
        self.ventana_principal = ventana_principal
        self.title("Procesamiento de Datos")

        estilo = ttk.Style(self)
        estilo.configure("CustomNotebook.TNotebook", tabposition='top')
        estilo.configure("CustomNotebook.TNotebook.Tab", padding=[35,12])

        self.archivos_procesamiento = {}   # Diccionario para archivos filtrados
        self.indices_personalizados = {}
        self.selected_for_processing = {}

        contenedor = tk.Frame(self)
        contenedor.pack(fill=tk.BOTH, expand=True)

        self.canvas_secundario = tk.Canvas(contenedor)
        self.canvas_secundario.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_vertical = tk.Scrollbar(contenedor, orient=tk.VERTICAL,
                                       command=self.canvas_secundario.yview)
        scroll_vertical.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_secundario.configure(yscrollcommand=scroll_vertical.set)

        self.marco_secundario = tk.Frame(self.canvas_secundario)
        self.canvas_secundario.create_window((0,0), window=self.marco_secundario, anchor="nw")
        self.marco_secundario.bind("<Configure>", lambda e:
            self.canvas_secundario.configure(scrollregion=self.canvas_secundario.bbox("all")))

        # Archivos Cargados
        tk.Label(self.marco_secundario, text="Archivos Cargados", font=("Arial",10,"bold")).pack(anchor="w", pady=5)
        marco_superior = tk.Frame(self.marco_secundario)
        marco_superior.pack(fill=tk.X, pady=5)

        self.lista_archivos_cargados = tk.Listbox(marco_superior, selectmode=tk.EXTENDED, height=10, width=30)
        self.lista_archivos_cargados.pack(side=tk.LEFT, fill=tk.Y)
        # Rellenar con los archivos que ya existen en la ventana principal
        for k,v in self.ventana_principal.archivos_datos.items():
            self.lista_archivos_cargados.insert(tk.END, v['nombre_mostrar'])

        marco_botones_lateral = tk.Frame(marco_superior)
        marco_botones_lateral.pack(side=tk.LEFT, padx=5)

        tk.Button(marco_botones_lateral, text="Añadir al Procesamiento",
                  command=self.anadir_al_procesamiento).pack(pady=2, anchor="w")
        tk.Button(marco_botones_lateral, text="Eliminar del Procesamiento",
                  command=self.eliminar_del_procesamiento).pack(pady=2, anchor="w")
        tk.Button(marco_botones_lateral, text="Añadir Todos al Procesamiento",
                  command=self.anadir_todos_al_procesamiento).pack(pady=2, anchor="w")
        tk.Button(marco_botones_lateral, text="Eliminar Todos del Procesamiento",
                  command=self.eliminar_todos_del_procesamiento).pack(pady=2, anchor="w")

        # Filtro
        marco_filtro = tk.Frame(self.marco_secundario)
        marco_filtro.pack(fill=tk.X, pady=5, anchor="w")

        tk.Label(marco_filtro, text="Filtro:", font=("Arial",10,"bold")).pack(side=tk.LEFT, padx=5)
        self.opciones_filtro = ["Ninguna","Media Móvil","Savitzky-Golay","Mediana","Gaussiano"]
        self.filtro_var = tk.StringVar(value=self.opciones_filtro[0])
        self.filtro_combobox = ttk.Combobox(marco_filtro, textvariable=self.filtro_var,
                                            values=self.opciones_filtro, state="readonly")
        self.filtro_combobox.pack(side=tk.LEFT, padx=5)

        tk.Button(marco_filtro, text="Aplicar Filtro", command=self.aplicar_filtro).pack(side=tk.LEFT, padx=5)

        # Tabla Filtrada
        marco_tabla_filtrada = tk.Frame(self.marco_secundario, width=600, height=200)
        marco_tabla_filtrada.pack_propagate(False)
        marco_tabla_filtrada.pack(fill=tk.BOTH, expand=True, pady=5)

        xscroll_filtrada = ttk.Scrollbar(marco_tabla_filtrada, orient=tk.HORIZONTAL)
        xscroll_filtrada.pack(side=tk.BOTTOM, fill=tk.X)
        yscroll_filtrada = ttk.Scrollbar(marco_tabla_filtrada, orient=tk.VERTICAL)
        yscroll_filtrada.pack(side=tk.RIGHT, fill=tk.Y)

        self.tabla_filtrada = ttk.Treeview(marco_tabla_filtrada, show="headings",
                                           xscrollcommand=xscroll_filtrada.set,
                                           yscrollcommand=yscroll_filtrada.set, height=8)
        self.tabla_filtrada.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_filtrada.config(command=self.tabla_filtrada.xview)
        yscroll_filtrada.config(command=self.tabla_filtrada.yview)

        tk.Button(self.marco_secundario, text="Exportar Datos Filtrados",
                  command=self.exportar_datos_filtrados).pack(pady=5)

        # Estadísticas Filtradas
        marco_estadisticas_filtradas = tk.Frame(self.marco_secundario, width=600, height=160)
        marco_estadisticas_filtradas.pack_propagate(False)
        marco_estadisticas_filtradas.pack(fill=tk.X, padx=5, pady=5)

        xscroll_estad_filtradas = ttk.Scrollbar(marco_estadisticas_filtradas, orient=tk.HORIZONTAL)
        xscroll_estad_filtradas.pack(side=tk.BOTTOM, fill=tk.X)

        self.tabla_estad_filtradas = ttk.Treeview(marco_estadisticas_filtradas, show="headings",
                                                  xscrollcommand=xscroll_estad_filtradas.set, height=7)
        self.tabla_estad_filtradas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_estad_filtradas.config(command=self.tabla_estad_filtradas.xview)

        # Gráfico con Filtro
        tk.Label(self.marco_secundario, text="Gráfico de Firma Espectral con Filtro",
                 font=("Arial",10,"bold")).pack(pady=5, anchor="w")
        marco_grafico_filtro = tk.Frame(self.marco_secundario, width=600, height=300)
        marco_grafico_filtro.pack_propagate(False)
        marco_grafico_filtro.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig_filtro = Figure(figsize=(6,4))
        self.eje_filtro = self.fig_filtro.add_subplot(111)
        self.eje_filtro.set_xlabel("Longitud de Onda")
        self.eje_filtro.set_ylabel("Reflectancia (Filtrada)")

        self.lienzo_filtro = FigureCanvasTkAgg(self.fig_filtro, master=marco_grafico_filtro)
        self.lienzo_filtro.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        marco_herramientas_filtro = tk.Frame(marco_grafico_filtro)
        marco_herramientas_filtro.pack(side=tk.BOTTOM, pady=10)
        self.barra_filtro = NavigationToolbar2Tk(self.lienzo_filtro, marco_herramientas_filtro)
        self.barra_filtro.update()
        
        # Notebook: Derivadas, Índices, Análisis
        self.notebook = ttk.Notebook(self.marco_secundario, style="CustomNotebook.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Pestaña Derivadas
        pestana_derivadas = tk.Frame(self.notebook)
        self.notebook.add(pestana_derivadas, text="Derivadas Espectrales")

        marco_tabla_deriv = tk.Frame(pestana_derivadas, width=600, height=200)
        marco_tabla_deriv.pack_propagate(False)
        marco_tabla_deriv.pack(fill=tk.BOTH, expand=True, pady=5)

        xscroll_deriv = ttk.Scrollbar(marco_tabla_deriv, orient=tk.HORIZONTAL)
        xscroll_deriv.pack(side=tk.BOTTOM, fill=tk.X)
        yscroll_deriv = ttk.Scrollbar(marco_tabla_deriv, orient=tk.VERTICAL)
        yscroll_deriv.pack(side=tk.RIGHT, fill=tk.Y)

        self.tabla_derivadas = ttk.Treeview(marco_tabla_deriv, show="headings",
                                            xscrollcommand=xscroll_deriv.set,
                                            yscrollcommand=yscroll_deriv.set, height=8)
        self.tabla_derivadas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_deriv.config(command=self.tabla_derivadas.xview)
        yscroll_deriv.config(command=self.tabla_derivadas.yview)

        marco_botones_deriv = tk.Frame(pestana_derivadas)
        marco_botones_deriv.pack(pady=5)
        tk.Button(marco_botones_deriv, text="Exportar Derivadas",
                  command=self.exportar_derivadas).pack(side=tk.LEFT, padx=5)

        self.fig_derivadas = Figure(figsize=(6,4))
        self.eje_derivadas = self.fig_derivadas.add_subplot(111)
        self.eje_derivadas.set_xlabel("Longitud de Onda")
        self.eje_derivadas.set_ylabel("Derivada Espectral")

        marco_grafico_deriv = tk.Frame(pestana_derivadas, width=600, height=300)
        marco_grafico_deriv.pack_propagate(False)
        marco_grafico_deriv.pack(fill=tk.BOTH, expand=True, pady=5)

        self.lienzo_derivadas = FigureCanvasTkAgg(self.fig_derivadas, master=marco_grafico_deriv)
        self.lienzo_derivadas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                

        # Crear un frame para la barra de herramientas de derivadas:
        marco_barra_derivadas = tk.Frame(marco_grafico_deriv)
        marco_barra_derivadas.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Barra de herramientas:
        self.barra_derivadas = NavigationToolbar2Tk(self.lienzo_derivadas, marco_barra_derivadas)
        self.barra_derivadas.update()


        self.eje_derivadas.legend(bbox_to_anchor=(1.05,1), loc="upper left", borderaxespad=0.)

        # Pestaña Índices
        pestana_indices = tk.Frame(self.notebook)
        self.notebook.add(pestana_indices, text="Índices de Vegetación")

        tk.Button(pestana_indices, text="Añadir Índice de Vegetación",
                  command=self.abrir_ventana_indice_personalizado).pack(pady=5, anchor="w")
        tk.Button(pestana_indices, text="Eliminar Índice Seleccionado",
                  command=self.eliminar_indice_de_lista).pack(pady=5, anchor="w")

        tk.Label(pestana_indices, text="Índices", font=("Arial",10,"bold")).pack(anchor="w", pady=2)

        self.tabla_indices = ttk.Treeview(pestana_indices, show="headings", selectmode="extended", height=5)
        self.tabla_indices.pack(fill=tk.X, padx=5, pady=5)
        self.tabla_indices["columns"] = ("Indice",)
        self.tabla_indices.heading("Indice", text="Indice")
        self.tabla_indices.column("Indice", width=100)

        self.indices_defecto = ["NDVI","EVI","SAVI","NBR","GLI","GCL","SIPI","MCARI"]
        for idx_n in self.indices_defecto:
            self.tabla_indices.insert("", tk.END, values=(idx_n,))

        tk.Label(pestana_indices, text="Índices Calculados", font=("Arial",10,"bold")).pack(anchor="w", pady=5)

        self.tabla_indices_calculados = ttk.Treeview(pestana_indices, show="headings", height=5)
        self.tabla_indices_calculados.pack(fill=tk.BOTH, padx=5, pady=5)

        tk.Button(pestana_indices, text="Exportar Índices",
                  command=self.exportar_indices_calculados).pack(pady=5)

        # Pestaña Análisis
        pestana_analisis = tk.Frame(self.notebook)
        self.notebook.add(pestana_analisis, text="Análisis")

        tk.Label(pestana_analisis, text="Archivos en Procesamiento",
                 font=("Arial",10,"bold")).pack(anchor="w", pady=5)

        self.lista_archivos_analisis = tk.Listbox(pestana_analisis, selectmode=tk.SINGLE, height=8, width=30)
        self.lista_archivos_analisis.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        tk.Button(pestana_analisis, text="Mostrar Análisis",
                  command=self.mostrar_analisis).pack(pady=5)

        self.texto_analisis = tk.Text(pestana_analisis, width=70, height=15, wrap=tk.WORD)
        self.texto_analisis.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.actualizar_lista_archivos_analisis()
        
        # -------- Pestaña Análisis (picos y valles)
        analysis_frame = tk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Análisis de picos y valles")

        analysis_table_frame = tk.Frame(analysis_frame, width=600, height=250)
        analysis_table_frame.pack_propagate(False)
        analysis_table_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        xscroll_analysis = ttk.Scrollbar(analysis_table_frame, orient=tk.HORIZONTAL)
        xscroll_analysis.pack(side=tk.BOTTOM, fill=tk.X)
        yscroll_analysis = ttk.Scrollbar(analysis_table_frame, orient=tk.VERTICAL)
        yscroll_analysis.pack(side=tk.RIGHT, fill=tk.Y)

        self.deriv_analysis_table = ttk.Treeview(
            analysis_table_frame, show='headings',
            xscrollcommand=xscroll_analysis.set,
            yscrollcommand=yscroll_analysis.set,
            height=10
        )
        self.deriv_analysis_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        xscroll_analysis.config(command=self.deriv_analysis_table.xview)
        yscroll_analysis.config(command=self.deriv_analysis_table.yview)

        # Columnas: Archivo | Pico/Valle | Longitud Onda | Valor Derivada
        self.deriv_analysis_table["columns"] = ("Archivo","Tipo","Longitud de Onda","Valor Derivada")
        for col in self.deriv_analysis_table["columns"]:
            self.deriv_analysis_table.heading(col, text=col)
            self.deriv_analysis_table.column(col, width=130)

        # Botón para analizar la derivada (picos y valles)
        self.analyze_button = tk.Button(analysis_frame, text="Analizar Derivada",
                                        command=self.analyze_derivative_peaks)
        self.analyze_button.pack(pady=5, anchor="w")

    # -------------------------------------------------------
    # Métodos para manejo de la lista de archivos
    # -------------------------------------------------------
    def anadir_al_procesamiento(self):
        sel = self.lista_archivos_cargados.curselection()
        if not sel:
            return
        nombres_sel = [self.lista_archivos_cargados.get(i) for i in sel]
        claves_sel  = [k for k,v in self.ventana_principal.archivos_datos.items()
                       if v['nombre_mostrar'] in nombres_sel]
        for c in claves_sel:
            if c not in self.archivos_procesamiento:
                self.archivos_procesamiento[c] = {
                    'longitudes': self.ventana_principal.archivos_datos[c]['longitudes'].copy(),
                    'reflectancias': self.ventana_principal.archivos_datos[c]['reflectancias'].copy(),
                    'nombre_mostrar': self.ventana_principal.archivos_datos[c]['nombre_mostrar'],
                    'reflectancias_originales': self.ventana_principal.archivos_datos[c]['reflectancias'].copy()
                }
        self.actualizar_tabla_filtrada()
        self.actualizar_lista_archivos_analisis()

    def eliminar_del_procesamiento(self):
        sel = self.lista_archivos_cargados.curselection()
        if not sel:
            return
        nombres_sel = [self.lista_archivos_cargados.get(i) for i in sel]
        claves_a_eliminar = [k for k,v in self.archivos_procesamiento.items()
                             if v['nombre_mostrar'] in nombres_sel]
        for cl in claves_a_eliminar:
            del self.archivos_procesamiento[cl]
        self.actualizar_tabla_filtrada()
        self.actualizar_lista_archivos_analisis()

    def anadir_todos_al_procesamiento(self):
        for k,v in self.ventana_principal.archivos_datos.items():
            if k not in self.archivos_procesamiento:
                self.archivos_procesamiento[k] = {
                    'longitudes': v['longitudes'].copy(),
                    'reflectancias': v['reflectancias'].copy(),
                    'nombre_mostrar': v['nombre_mostrar'],
                    'reflectancias_originales': v['reflectancias'].copy()
                }
        self.actualizar_tabla_filtrada()
        self.actualizar_lista_archivos_analisis()

    def eliminar_todos_del_procesamiento(self):
        self.archivos_procesamiento.clear()
        self.actualizar_tabla_filtrada()
        self.actualizar_lista_archivos_analisis()

    # -------------------------------------------------------
    # Filtros
    # -------------------------------------------------------
    def aplicar_filtro(self):
        if not self.archivos_procesamiento:
            messagebox.showwarning("Advertencia", "No hay archivos en el procesamiento.")
            return
        nombre_filtro = self.filtro_var.get()
        for c in self.archivos_procesamiento:
            if nombre_filtro == "Ninguna":
                self.archivos_procesamiento[c]['reflectancias'] = \
                    self.archivos_procesamiento[c]['reflectancias_originales'].copy()
            else:
                viejas_refl = self.archivos_procesamiento[c]['reflectancias']
                if nombre_filtro == "Media Móvil":
                    nuevas_refl = self.media_movil(viejas_refl, window=5)
                elif nombre_filtro == "Savitzky-Golay":
                    nuevas_refl = self.savitzky_golay(viejas_refl, window=5, poly=2)
                elif nombre_filtro == "Mediana":
                    nuevas_refl = self.mediana_filter(viejas_refl, window=3)
                elif nombre_filtro == "Gaussiano":
                    nuevas_refl = self.gaussiano_filter(viejas_refl, sigma=1)
                else:
                    nuevas_refl = viejas_refl
                self.archivos_procesamiento[c]['reflectancias'] = nuevas_refl

        self.actualizar_tabla_filtrada()
        self.actualizar_lista_archivos_analisis()

    def media_movil(self, datos, window=5):
        filtrada = []
        n= len(datos)
        mitad= window//2
        for i in range(n):
            inicio = max(0, i-mitad)
            fin    = min(n, i+mitad+1)
            filtrada.append(np.mean(datos[inicio:fin]))
        return np.array(filtrada)

    def savitzky_golay(self, datos, window=5, poly=2):
        return savgol_filter(datos, window, poly)

    def mediana_filter(self, datos, window=3):
        filtrada=[]
        n= len(datos)
        mitad= window//2
        for i in range(n):
            inicio = max(0, i-mitad)
            fin    = min(n, i+mitad+1)
            sub    = datos[inicio:fin]
            filtrada.append(np.median(sub))
        return np.array(filtrada)

    def gaussiano_filter(self, datos, sigma=1):
        return gaussian_filter1d(datos, sigma)

    def exportar_datos_filtrados(self):
        if len(self.tabla_filtrada["columns"])<=1:
            messagebox.showinfo("Info","No hay datos filtrados para exportar.")
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile="DatosFiltrados.txt",
                                            filetypes=[("Archivos de texto","*.txt")])
        if not ruta:
            return
        cols= self.tabla_filtrada["columns"]
        with open(ruta,'w') as f:
            f.write("\t".join(cols)+"\n")
            for row_id in self.tabla_filtrada.get_children():
                valores= self.tabla_filtrada.item(row_id,"values")
                fila_str= "\t".join(str(v) for v in valores)
                f.write(fila_str+"\n")
        messagebox.showinfo("Éxito","Datos filtrados exportados correctamente.")

    # -------------------------------------------------------
    # Tabla Filtrada + Estadísticas + Derivadas + Índices
    # -------------------------------------------------------
    def actualizar_tabla_filtrada(self):
        self.tabla_filtrada.delete(*self.tabla_filtrada.get_children())
        if not self.archivos_procesamiento:
            self.tabla_filtrada["columns"] = ("Longitud de Onda",)
            self.tabla_filtrada.heading("Longitud de Onda", text="Longitud de Onda")
            self.tabla_filtrada.column("Longitud de Onda", width=120)
            self.eje_filtro.clear()
            self.lienzo_filtro.draw()

            self.tabla_derivadas.delete(*self.tabla_derivadas.get_children())
            self.eje_derivadas.clear()
            self.lienzo_derivadas.draw()

            self.tabla_indices_calculados.delete(*self.tabla_indices_calculados.get_children())
            self.tabla_estad_filtradas.delete(*self.tabla_estad_filtradas.get_children())
            return

        archivos = list(self.archivos_procesamiento.keys())
        nombres_base = [self.archivos_procesamiento[a]['nombre_mostrar'] for a in archivos]
        cols = ["Longitud de Onda"] + nombres_base
        self.tabla_filtrada["columns"] = cols
        for c in cols:
            self.tabla_filtrada.heading(c, text=c)
            self.tabla_filtrada.column(c, width=120)

        primer_archivo = archivos[0]
        ondas_base = self.archivos_procesamiento[primer_archivo]['longitudes']

        for i, onda in enumerate(ondas_base):
            fila = [onda]
            for a in archivos:
                fila.append(self.archivos_procesamiento[a]['reflectancias'][i])
            self.tabla_filtrada.insert("", tk.END, values=tuple(fila))

        self.actualizar_grafico_filtrado()
        self.calcular_derivadas()
        self.calcular_indices_vegetacion()
        self.actualizar_tabla_estad_filtradas()

    def actualizar_grafico_filtrado(self):
        self.eje_filtro.clear()
        self.eje_filtro.set_xlabel("Longitud de Onda")
        self.eje_filtro.set_ylabel("Reflectancia (Filtrada)")

        if not self.archivos_procesamiento:
            self.lienzo_filtro.draw()
            return

        colores= ["red","blue","green","orange","purple","brown","cyan","magenta"]
        archivos= list(self.archivos_procesamiento.keys())
        for i,a in enumerate(archivos):
            ondas = self.archivos_procesamiento[a]['longitudes']
            refls = self.archivos_procesamiento[a]['reflectancias']
            nom   = self.archivos_procesamiento[a]['nombre_mostrar']
            c     = colores[i% len(colores)]
            self.eje_filtro.plot(ondas, refls, color=c, label=nom)

        self.eje_filtro.legend()
        self.lienzo_filtro.draw()

    def actualizar_tabla_estad_filtradas(self):
        self.tabla_estad_filtradas.delete(*self.tabla_estad_filtradas.get_children())
        etiqueta_stats = [
            "Reflectancia Mínima",
            "Longitud de Onda de Reflectancia Mínima",
            "Promedio",
            "Reflectancia Máxima",
            "Longitud de Onda de Reflectancia Máxima",
            "Desviación Estándar"
        ]
        if not self.archivos_procesamiento:
            self.tabla_estad_filtradas["columns"] = ("Estadística",)
            self.tabla_estad_filtradas.heading("Estadística", text="Estadística")
            self.tabla_estad_filtradas.column("Estadística", width=200)
            for lbl in etiqueta_stats:
                self.tabla_estad_filtradas.insert("", tk.END, values=(lbl,))
            return

        archivos = list(self.archivos_procesamiento.keys())
        nombres_base = [self.archivos_procesamiento[a]['nombre_mostrar'] for a in archivos]
        cols = ["Estadística"] + nombres_base + ["promedio"]
        self.tabla_estad_filtradas["columns"] = cols
        for c in cols:
            self.tabla_estad_filtradas.heading(c, text=c)
            w= 200 if c=="Estadística" else 120
            self.tabla_estad_filtradas.column(c, width=w)

        primer_archivo = archivos[0]
        ondas = self.archivos_procesamiento[primer_archivo]['longitudes']
        todos_refl = [self.archivos_procesamiento[a]['reflectancias'] for a in archivos]

        proms = [sum(r[i] for r in todos_refl)/ len(todos_refl)
                 for i in range(len(ondas))]

        def calcular_stats(refs, wls):
            r_min= min(refs)
            idx_mi= refs.tolist().index(r_min)
            wl_min= wls[idx_mi]
            r_max= max(refs)
            idx_ma= refs.tolist().index(r_max)
            wl_max= wls[idx_ma]
            r_mean= np.mean(refs)
            var= np.mean((refs-r_mean)**2)
            std_dev= math.sqrt(var)
            return (r_min, wl_min, r_mean, r_max, wl_max, std_dev)

        lista_stats= [calcular_stats(r, ondas) for r in todos_refl]
        stats_prom = calcular_stats(np.array(proms), ondas)

        for i, lbl in enumerate(etiqueta_stats):
            fila = [lbl]
            for st in lista_stats:
                fila.append(f"{st[i]:.4f}")
            fila.append(f"{stats_prom[i]:.4f}")
            self.tabla_estad_filtradas.insert("", tk.END, values=tuple(fila))

    # -------------------------------------------------------
    # Derivadas
    # -------------------------------------------------------
    def calcular_derivadas(self):
        self.tabla_derivadas.delete(*self.tabla_derivadas.get_children())
        self.eje_derivadas.clear()
        self.eje_derivadas.set_xlabel("Longitud de Onda")
        self.eje_derivadas.set_ylabel("Derivada Espectral")

        archivos= list(self.archivos_procesamiento.keys())
        if not archivos:
            self.lienzo_derivadas.draw()
            return

        ondas = self.archivos_procesamiento[archivos[0]]['longitudes']
        columnas = ["Longitud de Onda"] + [self.archivos_procesamiento[a]['nombre_mostrar'] for a in archivos]
        self.tabla_derivadas["columns"] = columnas
        for col in columnas:
            self.tabla_derivadas.heading(col, text=col)
            self.tabla_derivadas.column(col, width=120)

        # Crear filas en la tabla
        for i, onda in enumerate(ondas):
            fila = [onda] + [None]* len(archivos)
            self.tabla_derivadas.insert("", tk.END, values=fila)

        colores = ["red","blue","green","orange","purple","brown","cyan","magenta"]
        for i,a in enumerate(archivos):
            reflecs = self.archivos_procesamiento[a]['reflectancias']
            deriv   = np.gradient(reflecs, ondas) if len(reflecs)==len(ondas) else np.zeros_like(ondas)
            c       = colores[i% len(colores)]
            etiqueta= f"Derivada {self.archivos_procesamiento[a]['nombre_mostrar']}"
            self.eje_derivadas.plot(ondas, deriv, color=c, label=etiqueta)

            filas = self.tabla_derivadas.get_children()
            for j, fid in enumerate(filas):
                valores = list(self.tabla_derivadas.item(fid,"values"))
                valores[i+1] = deriv[j]
                self.tabla_derivadas.item(fid, values=tuple(valores))

        self.eje_derivadas.legend(bbox_to_anchor=(1.05,1), loc="upper left", borderaxespad=0.)
        self.lienzo_derivadas.draw()

    def exportar_derivadas(self):
        if len(self.tabla_derivadas["columns"])<=1:
            messagebox.showinfo("Info","No hay datos de derivadas para exportar.")
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile="Derivadas.txt",
                                            filetypes=[("Archivos de texto","*.txt")])
        if not ruta:
            return
        cols = self.tabla_derivadas["columns"]
        with open(ruta,'w') as f:
            f.write("\t".join(cols)+"\n")
            for row_id in self.tabla_derivadas.get_children():
                valores = self.tabla_derivadas.item(row_id,"values")
                fila_str= "\t".join(str(v) for v in valores)
                f.write(fila_str+"\n")
        messagebox.showinfo("Éxito","Derivadas exportadas correctamente.")

    # -------------------------------------------------------
    # Índices de Vegetación
    # -------------------------------------------------------
    def abrir_ventana_indice_personalizado(self):
        def al_añadir(nombre, formula, consts):
            self.indices_personalizados[nombre] = {
                "formula": formula,
                "consts": consts
            }
            self.tabla_indices.insert("", tk.END, values=(nombre,))
            self.calcular_indices_vegetacion()
        VentanaIndicePersonalizado(self, al_añadir_callback=al_añadir)

    def eliminar_indice_de_lista(self):
        sel = self.tabla_indices.selection()
        if not sel:
            return
        se_elimino = False
        bloqueado   = False
        for item_id in reversed(sel):
            indice_nom = self.tabla_indices.item(item_id,"values")[0]
            if indice_nom in self.indices_defecto:
                bloqueado = True
            else:
                self.tabla_indices.delete(item_id)
                se_elimino = True
                if indice_nom in self.indices_personalizados:
                    del self.indices_personalizados[indice_nom]

        if se_elimino:
            messagebox.showinfo("Info","Índices personalizados eliminados.")
        if bloqueado:
            messagebox.showwarning("Operación no permitida",
                                   "No se pueden eliminar índices clásicos (NDVI, EVI, etc.).")

        self.calcular_indices_vegetacion()

    def calcular_indices_vegetacion(self):
        self.tabla_indices_calculados.delete(*self.tabla_indices_calculados.get_children())
        if not self.archivos_procesamiento:
            return

        lista_indices = [self.tabla_indices.item(i,"values")[0]
                         for i in self.tabla_indices.get_children()]
        archivos = list(self.archivos_procesamiento.keys())
        nombres_base = [self.archivos_procesamiento[a]['nombre_mostrar'] for a in archivos]

        columnas = ["Índice"] + nombres_base
        self.tabla_indices_calculados["columns"] = columnas
        for col in columnas:
            self.tabla_indices_calculados.heading(col, text=col)
            self.tabla_indices_calculados.column(col, width=100)

        for indice in lista_indices:
            fila_valores = [indice]
            if indice in self.indices_defecto:
                for a in archivos:
                    lon  = self.archivos_procesamiento[a]['longitudes']
                    refl = self.archivos_procesamiento[a]['reflectancias']
                    # Buscar reflectancias en bandas clave
                    rojo    = obtener_reflectancia_en_onda(lon, refl, 670, 2.0)
                    verde   = obtener_reflectancia_en_onda(lon, refl, 550, 2.0)
                    azul    = obtener_reflectancia_en_onda(lon, refl, 470, 2.0)
                    nir     = obtener_reflectancia_en_onda(lon, refl, 860, 2.0)
                    swir2   = obtener_reflectancia_en_onda(lon, refl, 2200, 5.0)
                    val_ind = None
                    try:
                        if   indice=="NDVI" and (nir and rojo):
                            val_ind = calcular_ndvi(nir, rojo)
                        elif indice=="EVI" and (nir and rojo and azul):
                            val_ind = calcular_evi(nir, rojo, azul)
                        elif indice=="SAVI" and (nir and rojo):
                            val_ind = calcular_savi(nir, rojo)
                        elif indice=="NBR" and (nir and swir2):
                            val_ind = calcular_nbr(nir, swir2)
                        elif indice=="GLI" and (verde and rojo and azul):
                            val_ind = calcular_gli(verde, rojo, azul)
                        elif indice=="GCL" and (nir and verde):
                            val_ind = calcular_gcl(nir, verde)
                        elif indice=="SIPI" and (nir and azul and rojo):
                            val_ind = calcular_sipi(nir, azul, rojo)
                        elif indice=="MCARI" and (rojo and verde and nir):
                            val_ind = calcular_mcari(rojo, verde, nir)
                    except:
                        val_ind = None
                    fila_valores.append(f"{val_ind:.4f}" if val_ind is not None else "N/A")
            else:
                # Índice personalizado
                info_ind = self.indices_personalizados.get(indice)
                if not info_ind:
                    for _a in archivos:
                        fila_valores.append("N/A")
                else:
                    formula = info_ind["formula"]
                    consts  = info_ind["consts"]
                    for a in archivos:
                        lon  = self.archivos_procesamiento[a]['longitudes']
                        refl = self.archivos_procesamiento[a]['reflectancias']
                        val_ = calcular_indice_personalizado(formula, consts, lon, refl)
                        fila_valores.append(f"{val_:.4f}" if val_ is not None else "N/A")

            self.tabla_indices_calculados.insert("", tk.END, values=tuple(fila_valores))

    def exportar_indices_calculados(self):
        if len(self.tabla_indices_calculados["columns"]) <= 1:
            messagebox.showinfo("Info","No hay índices calculados para exportar.")
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".txt",
                                            initialfile="IndicesCalculados.txt",
                                            filetypes=[("Archivos de texto","*.txt")])
        if not ruta:
            return
        cols = self.tabla_indices_calculados["columns"]
        with open(ruta,'w') as f:
            f.write("\t".join(cols)+"\n")
            for fila_id in self.tabla_indices_calculados.get_children():
                valores = self.tabla_indices_calculados.item(fila_id,"values")
                fila_str= "\t".join(str(x) for x in valores)
                f.write(fila_str+"\n")
        messagebox.showinfo("Éxito","Índices Calculados exportados correctamente.")

    # -------------------------------------------------------
    # Análisis
    # -------------------------------------------------------
    def actualizar_lista_archivos_analisis(self):
        self.lista_archivos_analisis.delete(0, tk.END)
        for c in self.archivos_procesamiento:
            nom = self.archivos_procesamiento[c]['nombre_mostrar']
            self.lista_archivos_analisis.insert(tk.END, nom)

    def mostrar_analisis(self):
        self.texto_analisis.delete("1.0", tk.END)
        sel = self.lista_archivos_analisis.curselection()
        if not sel:
            self.texto_analisis.insert(tk.END, "No se ha seleccionado ningún archivo.\n")
            return
        nombre_sel = self.lista_archivos_analisis.get(sel[0])

        clave = None
        for k,v in self.archivos_procesamiento.items():
            if v['nombre_mostrar'] == nombre_sel:
                clave = k
                break
        if clave is None:
            self.texto_analisis.insert(tk.END, "Error: archivo no encontrado en el procesamiento.\n")
            return

        ondas = self.archivos_procesamiento[clave]['longitudes']
        refls = self.archivos_procesamiento[clave]['reflectancias']

        azul_   = obtener_reflectancia_en_onda(ondas, refls, 470, 2.0)
        verde_  = obtener_reflectancia_en_onda(ondas, refls, 550, 2.0)
        rojo_   = obtener_reflectancia_en_onda(ondas, refls, 670, 2.0)
        nir_    = obtener_reflectancia_en_onda(ondas, refls, 860, 2.0)
        swir2_  = obtener_reflectancia_en_onda(ondas, refls, 2200, 5.0)

        estadisticas_filtradas = self.obtener_estadisticas_filtradas_de_archivo(nombre_sel)

        # Derivadas
        max_deriv_onda = None
        columnas_deriv = list(self.tabla_derivadas["columns"])
        if nombre_sel in columnas_deriv:
            idx_col = columnas_deriv.index(nombre_sel)
            todas_filas = self.tabla_derivadas.get_children()
            maxima_deriv = None
            onda_max     = None
            for fid in todas_filas:
                vals= self.tabla_derivadas.item(fid, "values")
                try:
                    onda_val = float(vals[0])
                    dval_str = vals[idx_col]
                    dval_    = float(dval_str)
                    if (maxima_deriv is None) or (dval_ > maxima_deriv):
                        maxima_deriv = dval_
                        onda_max     = onda_val
                except:
                    pass
            max_deriv_onda = onda_max

        # Índices
        columnas_idx= list(self.tabla_indices_calculados["columns"])
        if nombre_sel not in columnas_idx:
            self.texto_analisis.insert(tk.END,
                f"Archivo: {nombre_sel}\nNo hay índices calculados.\n")
            return
        idx_archivo= columnas_idx.index(nombre_sel)
        valores_indices= {}
        for fila_id in self.tabla_indices_calculados.get_children():
            vals = self.tabla_indices_calculados.item(fila_id,"values")
            nom_indice = vals[0]
            if idx_archivo< len(vals):
                try:
                    val_f = float(vals[idx_archivo])
                except:
                    val_f = None
                valores_indices[nom_indice] = val_f
        
                
        
        
        texto_final = self.generar_texto_analisis(nombre_sel, valores_indices,
                                                  (azul_, verde_, rojo_, nir_, swir2_),
                                                  estadisticas_filtradas, max_deriv_onda)
        self.texto_analisis.insert(tk.END, texto_final)


    def obtener_estadisticas_filtradas_de_archivo(self, nombre_archivo):
        """
        Retorna un diccionario con los valores de estadística
        (Mín, Máx, Promedio, etc.) para el 'nombre_archivo'
        basados en la tabla de estadísticas filtradas.
        Si el archivo no aparece en las columnas, se retorna {}.
        """
        fila_ids = self.tabla_estad_filtradas.get_children()
        columnas_estad = self.tabla_estad_filtradas["columns"]
        if nombre_archivo not in columnas_estad:
            return {}
        
        idx_col = columnas_estad.index(nombre_archivo)
        dicc_stats = {}

        # Recorrer cada fila de la tabla de estadísticas filtradas
        for fila_id in fila_ids:
            valores = self.tabla_estad_filtradas.item(fila_id, "values")
            nombre_estad = valores[0]  # Primera columna: "Reflectancia Mínima", etc.
            if idx_col < len(valores):
                # El valor de estadística para este archivo se encuentra en la posición idx_col
                dicc_stats[nombre_estad] = valores[idx_col]
        
        return dicc_stats
    
    def interpretar_indice(self, nombre_indice, val):
        """
        Retorna mensajes interpretativos simples para cada índice,
        basándose en umbrales básicos.
        """
        msgs=[]
        if nombre_indice=="NDVI":
            if val<0:
                msgs.append("   => NDVI negativo: superficies no vegetadas (agua, etc.).")
            elif val<0.2:
                msgs.append("   => NDVI muy bajo: vegetación muy escasa o suelos descubiertos.")
            elif val<0.5:
                msgs.append("   => NDVI moderado: vegetación en desarrollo o densidad media.")
            else:
                msgs.append("   => NDVI alto: vegetación densa y saludable.")

        elif nombre_indice=="EVI":
            if val<0.1:
                msgs.append("   => EVI muy bajo: poca vegetación.")
            elif val<0.3:
                msgs.append("   => EVI bajo/moderado.")
            else:
                msgs.append("   => EVI alto: alta densidad de follaje.")

        elif nombre_indice=="SAVI":
            if val<0.1:
                msgs.append("   => SAVI muy bajo: escasa vegetación o suelo expuesto.")
            elif val<0.3:
                msgs.append("   => SAVI moderado.")
            else:
                msgs.append("   => SAVI alto: buena cobertura vegetal (corrige el efecto del suelo).")

        elif nombre_indice=="GLI":
            if val<0:
                msgs.append("   => GLI negativo: refleja poco en verde comparado con rojo/azul.")
            elif val<0.2:
                msgs.append("   => GLI bajo: poca intensidad en banda verde.")
            else:
                msgs.append("   => GLI alto: vegetación intensa en banda verde.")

        elif nombre_indice=="GCL":
            if val<0:
                msgs.append("   => GCL negativo: banda green > NIR => poca clorofila.")
            elif val<0.5:
                msgs.append("   => GCL moderado: cierta clorofila, no muy alta.")
            else:
                msgs.append("   => GCL alto: alta concentración de clorofila.")

        elif nombre_indice=="SIPI":
            if val<1.0:
                msgs.append("   => SIPI bajo: menor pigmentación carotenoide o alta absorción en azul.")
            else:
                msgs.append("   => SIPI alto: alta diferencia (nir-blue)/(nir-red), más carotenoides.")

        elif nombre_indice=="MCARI":
            if val<0.3:
                msgs.append("   => MCARI bajo: baja absorción en rojo => poca clorofila.")
            else:
                msgs.append("   => MCARI alto: fuerte absorción roja => alta clorofila.")

        elif nombre_indice=="NBR":
            if val<0.1:
                msgs.append("   => NBR bajo: vegetación quemada o suelos desnudos.")
            else:
                msgs.append("   => NBR moderado/alto: vegetación no quemada.")

        return msgs
    
    def generar_texto_analisis(self, nombre_archivo, valores_indices,
                               reflectancias_clave, estadisticas_filtradas, max_deriv_onda):
        """
        Construye un texto final de análisis, combinando:
          - Reflectancias clave
          - Estadísticas filtradas
          - Máxima derivada
          - Índices de vegetación y su interpretación
          - Relación entre ciertos índices (NDVI/EVI/SAVI, GCL/MCARI)
        """
        azul_, verde_, rojo_, nir_, swir2_ = reflectancias_clave
        lineas = []
        lineas.append(f"Análisis del archivo: {nombre_archivo}\n")
        
        # 1. Mostrar reflectancias clave (aprox)
        lineas.append("Reflectancias clave (aprox):")
        def imprimir_o_nd(etiqueta, valor):
            if valor is None:
                return f" - {etiqueta}: N/D"
            else:
                return f" - {etiqueta}: {valor:.4f}"

        lineas.append(imprimir_o_nd("Azul (470 nm)", azul_))
        lineas.append(imprimir_o_nd("Verde (550 nm)", verde_))
        lineas.append(imprimir_o_nd("Rojo (670 nm)",  rojo_))
        lineas.append(imprimir_o_nd("NIR (860 nm)",   nir_))
        lineas.append(imprimir_o_nd("SWIR2(2200 nm)", swir2_))
        lineas.append("")

        # 2. Estadísticas filtradas
        if estadisticas_filtradas:
            lineas.append("Estadísticas (basadas en datos filtrados):")
            orden_estad = [
                "Reflectancia Mínima",
                "Longitud de Onda de Reflectancia Mínima",
                "Promedio",
                "Reflectancia Máxima",
                "Longitud de Onda de Reflectancia Máxima",
                "Desviación Estándar"
            ]
            for clave_estad in orden_estad:
                valor_estad = estadisticas_filtradas.get(clave_estad, "N/A")
                lineas.append(f" - {clave_estad}: {valor_estad}")
            lineas.append("")

        # 3. Derivada (posición de máxima derivada)
        if max_deriv_onda is not None:
            lineas.append(f"Máxima Derivada ~ {max_deriv_onda:.2f} nm.")
            if 680 <= max_deriv_onda <= 730:
                lineas.append(" => Indica un 'red edge' típico de vegetación sana.\n")
            else:
                lineas.append(" => El 'red edge' no está en el rango usual (680-730 nm).\n")
        else:
            lineas.append("No se encontró información de la derivada para este archivo.\n")

        # 4. Índices Calculados
        lineas.append("Índices Calculados:\n")
        for indice_basico in ["NDVI","EVI","SAVI","NBR","GLI","GCL","SIPI","MCARI"]:
            valor_indice = valores_indices.get(indice_basico, None)
            if valor_indice is not None:
                lineas.append(f" - {indice_basico}: {valor_indice:.4f}")
                # Agregar interpretación
                lineas.extend(self.interpretar_indice(indice_basico, valor_indice))
        
        lineas.append("")

        # 5. Relación entre índices (NDVI/EVI/SAVI) y (GCL/MCARI)
        ndvi = valores_indices.get("NDVI", None)
        evi  = valores_indices.get("EVI", None)
        savi = valores_indices.get("SAVI", None)
        gcl  = valores_indices.get("GCL", None)
        mcari= valores_indices.get("MCARI", None)
        # NDVI, EVI, SAVI
        if ndvi is not None and evi is not None and savi is not None:
            if (ndvi > 0.5) and (evi < 0.3 or savi < 0.3):
                lineas.append("Atención: NDVI alto, pero EVI o SAVI bajos => posible saturación o suelo.\n")
            elif (ndvi < 0.2) and (evi > 0.4 or savi > 0.4):
                lineas.append("Atención: NDVI bajo, pero EVI o SAVI altos => incongruencia en bandas.\n")
            else:
                lineas.append("Los índices NDVI, EVI y SAVI no presentan contradicciones significativas.\n")

        # GCL/MCARI (clorofila)
        if gcl is not None and mcari is not None:
            if gcl > 0.5 and mcari < 0.3:
                lineas.append("Atención: GCL alto, MCARI bajo => señales conflictivas sobre la clorofila.\n")
            elif gcl < 0.2 and mcari > 0.3:
                lineas.append("Atención: GCL bajo, MCARI alto => absorción roja alta pero reflectancia NIR/verde no concuerda.\n")
            else:
                lineas.append("GCL y MCARI no presentan contradicciones fuertes sobre clorofila.\n")

        lineas.append("")
        return "\n".join(lineas)
    
    def analyze_derivative_peaks(self):
       from scipy.signal import find_peaks

       # 1) limpiar la tabla de resultados
       self.deriv_analysis_table.delete(*self.deriv_analysis_table.get_children())

       # 2) comprobar que hay archivos listos
       if not self.archivos_procesamiento:
           messagebox.showwarning("Advertencia",
                                  "No hay archivos en el procesamiento.")
           return

       for ruta, info in self.archivos_procesamiento.items():
           wl   = info['longitudes']       # longitudes de onda
           refl = info['reflectancias']    # reflectancias
           nombre = info['nombre_mostrar'] # nombre para mostrar

           if len(wl) < 2 or len(wl) != len(refl):
               continue

           # 3) derivada numérica
           deriv = np.gradient(refl, wl)

           # 4) buscar picos y valles de la derivada
           peaks,   _ = find_peaks( deriv, prominence=1e-6)
           valleys, _ = find_peaks(-deriv, prominence=1e-6)

           # 5) volcar resultados a la tabla
           for idx in peaks:
               self.deriv_analysis_table.insert(
                   "", tk.END,
                   values=(nombre, "Pico",
                           f"{wl[idx]:.2f}", f"{deriv[idx]:.5f}")
               )

           for idx in valleys:
               self.deriv_analysis_table.insert(
                   "", tk.END,
                   values=(nombre, "Valle",
                           f"{wl[idx]:.2f}", f"{deriv[idx]:.5f}")
               )

       messagebox.showinfo("Análisis de Derivadas",
                           "Proceso finalizado. Revisa la tabla de picos y valles.")


# -------------------------------------------------------------------------
# EJECUCIÓN DE CODIGO
# -------------------------------------------------------------------------
if __name__ == "__main__":
    raiz = tk.Tk()
    aplicacion = VentanaPrincipal(raiz)
    raiz.mainloop()
