# BlackHoles
Кодът на последната задача е в папка kerr_numeric.

Та за момента имам:
  - Функции, които смятат първа производна на функция с една/много променливи и втора производна на много променливи. Всички използват дуални и хипердуални числа, което позволява пресмятането на производни с машинна точност.
    
    - derivative(func: Callable, x: float) ->  float
    
    - partial_deriv(func: Callable, vars: Union[list, np.ndarray], wrt_index: int, *params) -> float
    
    - second_deriv(func: Callable, x: float, *params) -> float
    
    - second_partial_deriv(func: Callable, vars: Union[list, np.ndarray], wrt_index: Union[list, np.ndarray], *params) -> float
    
  - Две функции за пресмятане на якобиана - единият използва като вход лист от новите координати като функции на старите за пресмятане, другия изпозва хамилтониана:
    
    - jacobian(new_coord: List[Callable], old_coord: Union[list, np.ndarray], *params) -> np.ndarray
    
    - jacobian_H(H, qp, *params) -> 8x8 np.ndarray (може да го напиша за по-общ случай, ако ми потрябва)
    
  - Симплектичен интегратор с възможности на интегриране от 2ри и 4ти ред:
    
    - symplectic_integrator(H, qp0, metric_params, step_size: float, omega: float, num_steps: int, ord: float) -> 2 * len(qp0) x num_steps np.ndarray
    
  - Функция за закръгляне на стойността на ъгъл, в зависимост от зададена резолюция и интервал:
    
    ++ (мързи ме да пиша сега
    - discretize_angles(angles: list, resolution: int, interval=np.pi) -> tuple с (\alpha, \delta \alpha)
    
  - ... и други подобни, които не се знае дали ще използвам някъде.
    
    (Това го казвам, защото тези функции могат да се използват за неща извън сегашната задача.)
