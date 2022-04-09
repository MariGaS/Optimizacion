
import numpy as np

def Rosembrock(x, params = {}): #función de Rosembrock
    '''
    Parámetros
    -----------
        n    : dimensión de muestra
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        f_x : Evaluación en la función de Rosembrock
    '''
    n  = x.shape[0]
    f_x = 0.0
    
    for i in range(n-1) :
        f_x += 100.0 * (x[i+1] - x[i]* x[i]) * (x[i+1] - x[i]* x[i])
        f_x += (1 - x[i])*(1 - x[i])
    
    return f_x

def grad_Rosembrock(x, params = {}): 
    '''
    Parámetros
    -----------
        n    : dimensión de muestra
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        f_x : Evaluación en la función de Rosembrock
    '''
    n  = x.shape[0]
    Grad_f = np.zeros(n, dtype = np.float64)
    
    if(n == 2): 
      #400x31 −400x1x2 −2 + 2x1
        Grad_f[0] = 400.0*( x[0]*x[0]*x[0] ) - 400.0 *x[0]*x[1] -2.0 + 2.0* x[0]
        Grad_f[1] = 200.0* x[1] - 200.0 *x[0]*x[0]
          
        return Grad_f 
        
    elif( n > 2): 
        
        Grad_f[0] = 400.0*( x[0]*x[0]*x[0] ) - 400.0 *x[0]*x[1] -2.0 + 2.0* x[0]
        Grad_f[n-1] = 200.0* x[n-1] - 200.0 *x[n-2]*x[n-2]
        
        for i in range(n-1): 
            Grad_f[i] = 400.0*(x[i]*x[i]*x[i]) + 202.0*x[i]-2.0-400.0*x[i]*x[i+1]-200* x[i-1]*x[i-1]
          
        return Grad_f

def hess_Rosembrock(x, params = {}): 
    '''
    Parámetros
    -----------
        n    : dimensión de muestra
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
       H_f : Evaluación del Hessiano de Rosembrock en el punto dado 
    '''
    n = x.shape[0]
    H_f = np.zeros((n,n), dtype = np.float64)
    
    for i in range(0,n-1) :
        H_f[i][i]   = 1200.0 * x[i] * x[i] - 400.0 * x[i+1] + 202.0
        H_f[i][i+1] = - 400.0 * x[i]
        H_f[i+1][i] = - 400.0 * x[i]
    H_f[0][0] = 1200.0*x[0]*x[0] +2.0 - 400.0 *x[1]
    H_f[n-1][n-1] = 200.0
    
    return H_f

def wood_f(x, params = {}): 
    '''
    Parámetros
    -----------
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        f_x : Evaluación en la función Wood
    '''
    f_x = 0.0
    f_x += 100.0 * (x[0]*x[0] - x[1]) * (x[0]*x[0] - x[1])
    f_x += (x[0]-1.0) * (x[0]-1.0)
    f_x += (x[2]-1.0) * (x[2]-1.0)
    f_x += 90.0 *  (x[2]*x[2]-x[3]) * (x[2]*x[2]-x[3])
    f_x += 10.1 * ((x[1]-1.0) * (x[1]-1.0) + (x[3]-1.0) * (x[3]-1.0)) 
    f_x += 19.8 * (x[1]-1.0) * (x[3]-1.0)
    
    return f_x

def grad_wood(x, params = {}):
    '''
    Parámetros
    -----------
        n    : dimensión de muestra
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
        Gr_wood : gradiente de Wood evaluado   
    ''' 
    n = x.shape[0]
    
    Gr_wood =  np.zeros(n, dtype = np.float64)
    
    Gr_wood[0] = 400.0 * (x[0]* x[0] - x[1]) * x[0] + 2.0 * (x[0] - 1.0)
    Gr_wood[1] = - 200.0 * (x[0]* x[0] - x[1]) + 20.2 * (x[1] - 1.0) + 19.8 * (x[3] - 1.0)
    Gr_wood[2] = 2.0 * (x[2] - 1.0) + 360.0 * (x[2]* x[2] - x[3]) * x[2]
    Gr_wood[3] = -180.0 * (x[2]* x[2] - x[3]) + 20.2 * (x[3] - 1.0) + 19.8 * (x[1] - 1)
    

    return Gr_wood

def hess_wood(x, params = {}): 
    '''
    Parámetros
    -----------
        n    : dimensión de muestra
        x    : vector de valores [x_1, x_2, ..., x_n]
    Regresa
    -----------
       H_f : Evaluación del Hessiano de Wood en el punto dado 
    '''
    n = x.shape[0]
    H_f = np.zeros((n,n), dtype = np.float64)
    
    H_f[0][0] = 1200.0 * x[0]* x[0] -400.0*x[1] +2.0
    H_f[0][1] = -400.0 * x[0]
    H_f[1][0] = -400.0 * x[0]
    H_f[1][1] = 220.2
    H_f[1][3] = 19.8 
    H_f[3][1] = 19.8 
    H_f[2][2] = 1080.0 * x[2]* x[2] -360.0 * x[3] + 2.0
    H_f[3][2] = -360.0 * x[2]
    H_f[2][3] = -360.0 * x[2]
    H_f[3][3] = 220.2
    
    return H_f

def branin_f( x, params = {}):
    '''
    Parámetros
    -----------
        x           : vector de valores [x_1, x_2]
        a,b,c,r,s,t : parámetros de la función
    Regresa
    -----------
       sum : Evaluación de la función branin en el punto dado 
    '''
    a = params['a']
    b = params['b']
    c = params['c']
    r = params['r']
    s = params['s']
    t = params['t']

    sum = 0 
    sum += a*(x[1]- b* (x[0]**2) + c*x[0] -r )*(x[1]- b* (x[0]**2) + c*x[0] -r )
    sum += s*(1-t)* np.cos(x[0])
    sum += s
    
    return  sum

def grad_branin(x, params ={}):
    '''
    Parámetros
    -----------
        x           : vector de valores [x_1, x_2]
        a,b,c,r,s,t : parámetros de la función
    Regresa
    -----------
       grad_f : Evaluación del gradiente de la función de branin en el punto dado 
    '''
    #parcial_x1 = 2a(x_2-bx_1^2 +cx1-r) *(-2bx_1+c) +s(t-1)sen(x_1) 
    #parcial_x2 = 2a(x_2-bx_1^2 +cx1-r)
    a = params['a']
    b = params['b']
    c = params['c']
    r = params['r']
    s = params['s']
    t = params['t']

    grad_f = np.zeros(2, dtype= np.float64)
    
    grad_f[0] = 2.0*a* (x[1]- b* (x[0]**2) + c*x[0] -r )*(-2.0*b*x[0]+c) +s*(t-1)*np.sin(x[0])
    grad_f[1] = 2.0 *a* (x[1]- b* (x[0]**2) + c*x[0] -r )

    return grad_f

def hess_branin(x, params={}):
    '''
    Parámetros
    -----------
        x           : vector de valores [x_1, x_2]
        a,b,c,r,s,t : parámetros de la función
    Regresa
    -----------
       H_f : Evaluación del hessiano de la función de branin en el punto dado 
    '''
    H_f = np.zeros((2,2), dtype = np.float64)
    
    a = params['a']
    b = params['b']
    c = params['c']
    r = params['r']
    s = params['s']
    t = params['t']
    
    H_f[0][0] = -4.0 * a * b * (x[1] - b * x[0]**2 + c * x[0] - r) 
    H_f[0][0] += 2.0 * a * (-2.0 * b * x[0] + c) **2 - s * (1.0 - t) * np.cos(x[1])

    H_f[0][1] = H_f[1][0] = 2.0 * a * ( -2.0 * b * x[0] + c)

    H_f[1][1] = 2.0 * a

    return H_f


def cholesky_modificado(A, beta = 1e-3, max_iter = 100):
    diag = np.diag(A)
    min_aii = min(diag)
    if min_aii > 0 :
        tau = 0.0
    else :
        tau = - min_aii + beta

    k = 0
    while k < max_iter :
        A = A + tau * np.eye(diag.shape[0])
        try :
            A = np.linalg.cholesky(A)
        except :
            tau = max(2.0 * tau, beta)
        else :
            k = max_iter
        k += 1

    return A



def bisection_wolf( x_k, d_k, fun = None, grad = None, alpha_k = 1.0, c1 = 1e-4, c2 = 0.9, max_iter = 100, params = {}):
    '''
    Bisection with wolfe conditions

    Parámetros
    ------------
    c1,c2        : constantes 
    alpha_k      : tamaño de paso en la iteración k de steepest descent 
    max_iter     : máximo número de iteraciones para encontrar el tamaño de paso 
    fun          : función a minimizar
    grad         : gradiente de la función  
    x_k          : punto en la iteración k del decenso de gradiente 
    d_k          : dirección de descenso 
    '''

    
    alpha    = 0 
    beta     = np.inf 
    alpha_i = alpha_k
    i = 0
    for j in range(max_iter):
        if  fun(x_k + alpha_i * d_k, params) > fun(x_k, params) + (c1*alpha_i)* grad(x_k, params).dot(d_k):
            beta     = alpha_i
            alpha_i  = 0.5 * (alpha + beta)
        elif grad(x_k+alpha_i*d_k, params).dot(d_k) < c2 * np.dot(grad(x_k, params), d_k):
            alpha = alpha_i 
            if beta == np.inf :
                alpha_i = 2.0 * alpha  
            else : 
                alpha_i = 0.5 * (alpha + beta)
        else: 
            break
    
    
    alpha_k = alpha_i
       
    return alpha_k  



def newton_modificado (params = []) :
    '''
    Método de Newt_modificado 
    '''
    # Cargo parámetros
    x_k        = params['x_0']
    x_k_next   = None
    f          = params['f']
    f_grad     = params['f_grad']
    f_hess     = params['f_hess']
    max_iter   = params['max_iter']
    tau_x      = params['tau_x']
    tau_f      = params['tau_f']
    tau_f_grad = params['tau_grad']
    beta       = params['beta']
    max_iter_c = params['cholesky']['max_iter']

    # Subpámetros para la función
    if f.__name__ == 'branin_f' :
        sub_params = {
                    'a' : params['a'],
                    'b' : params['b'],
                    'c' : params['c'],
                    'r' : params['r'],
                    's' : params['s'],
                    't' : params['t']
                  }
    else :
        sub_params = {}
        
    
    f_history = []
    f_history.append(f(x_k, params = sub_params))

    g_history = []
    g_history.append(np.linalg.norm(f_grad(x_k, params = sub_params)))
              
    
    k = 0 
    
    while True:
        # gradiente 
        g_k = f_grad(x_k, params = sub_params)
        # hessiano 
        B_k = f_hess(x_k, params = sub_params)
        # factorización usando cholesky modificado 
        L = cholesky_modificado(B_k, beta, max_iter_c)   
        #Resolver el sistema B_kd_k = -g_k 
        y   = np.linalg.solve(L, -g_k) 
        d_k = np.linalg.solve(L.T, y)
        
        #primero obtenemos alpha 
        
        alpha = bisection_wolf( x_k, d_k, fun = f, grad = f_grad, alpha_k = 1.0, c1 = 1e-4, c2 = 0.9, max_iter = 100, params = sub_params)
        # Acutalizamos el siguiente x 
        x_k_next = x_k + alpha * d_k   
        
        # Guardo Parámetros
        f_history.append(f(x_k_next, sub_params))
        g_history.append(np.linalg.norm(f_grad(x_k_next, sub_params)))
        
                  
        # Criterios de paro
        if (k > max_iter) :
            break
            
        if np.linalg.norm(x_k_next - x_k)/max(np.linalg.norm(x_k), 1.0) < tau_x :
            break
                  
        if np.abs(f(x_k_next, sub_params) - f(x_k, sub_params)) / max(np.linalg.norm(f(x_k, sub_params)), 1.0) < tau_f :
            break
        
                  
        if np.linalg.norm(f_grad(x_k_next, sub_params)) < tau_f_grad :
            break
            
           
        x_k = x_k_next       
        k   = k + 1
        
    return np.array(f_history), np.array(g_history), x_k_next