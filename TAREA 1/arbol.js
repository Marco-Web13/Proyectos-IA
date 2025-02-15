class Nodo {
    constructor(dato) {
        this.dato = dato;
        this.izquierda = null;
        this.derecha = null;
    }
}
      
    function agregar(nodo, dato) {
        if (nodo === null) {
            return new Nodo(dato);
        }
        if (dato < nodo.dato) {
            nodo.izquierda = agregar(nodo.izquierda, dato); // Recursión hacia la izquierda
        } else {
            nodo.derecha = agregar(nodo.derecha, dato); // Recursión hacia la derecha
        }
        return nodo; // Devuelve el nodo actual
    }

    function preorden(nodo) {
        if (nodo !== null) {
            console.log(nodo.dato + ", ");
            preorden(nodo.izquierda);
            preorden(nodo.derecha);
        }
    }

let raiz = null
raiz = agregar(raiz, 10);
raiz = agregar(raiz, 5);
raiz = agregar(raiz, 15);
raiz = agregar(raiz, 3);
raiz = agregar(raiz, 7);

preorden(raiz);