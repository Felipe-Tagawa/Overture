## Contexto Astronômico

> Seção explicando o significado físico das features orbitais utilizadas e do target,
> para leitores sem background em astronomia/mecânica orbital.

### O que descreve uma órbita?

Para introduzirmos a análise que será feita nesse projeto, é fundamental entendermos o funcionamento de uma órbita. Nessa caso, estamos tratando de órbitas de asteroides, que são corpos celestes que orbitam o Sol. Uma órbita elíptica tradicional em três dimensões no espaço podem ser descritas por um conjunto de 6 parâmetros conhecidos como elementos orbitais keplerianos.

Em nossos estudos utilizaremos 5 desses elementos orbitais como features para prever o MOID (Minimum Orbit Intersection Distance), que é a menor distância entre a órbita do asteroide e a órbita da Terra. O MOID é uma métrica importante para avaliar o risco de impacto de um asteroide com a Terra. O 6° elemento orbital, anomalia média, não será utilizado como feature, pois ele descreve a posição do asteroide em sua órbita em um dado instante de tempo, e não influencia na geometria da órbita em si, ou seja, não influencia no MOID.

Para melhor compreensão, analisaremos a órbita como camadas. Cada um dos elementos descreve uma característica e a combinação deles determina a geometria final da órbita.

1. Forma da Elipse: descrita pelos elementos `a` (semi-eixo maior) e `e` (excentricidade)
2. Orientação do plano orbital no espaço: descrita pelos elementos `i` (inclinação) e `om` (longitude do nó ascendente)
3. Orientação da elipse dentro do plano orbital: descrita pelo elemento `w` (argumento do perihélio)

#### Leis de Kepler

1° Lei de Kepler: A órbita de um planeta é uma elipse com o Sol em um dos focos.
2° Lei de Kepler: A linha que liga um planeta ao Sol varre áreas iguais em tempos iguais.
3° Lei de Kepler: O quadrado do período orbital de um planeta é proporcional ao cubo do semi-eixo maior da sua órbita.

#### Features orbitais utilizadas

1° `a` — Semi-eixo maior: descreve o tamanho da órbita elíptica, ou seja, a distância média do asteroide ao Sol. Quanto maior o semi-eixo maior, mais distante do Sol o asteroide está.

Interpretação prática: `a` classifica onde o asteroide se encontra no sistema solar, valores próximos de 1 UA (1 unidade astronômica, a distância média da Terra ao Sol) indicam órbitas que cruzam ou se aproximam da órbita da Terra (NEAs - Near-Earth Asteroids); Valores médios de 2-4 UA indicam asteroides do cinturão principal, enrte marte e júpiter. Por isso, no código filtramos valroes de `a` muito grandes (utilizamos a < 100 UA) para evitar cometas de órbitas quase parabólicas, que não são o foco do estudo, haja vista que cometas de períodos longos posssuem órbitas muito alongadas e podem ter MOIDs muito pequenos, mas não representam risco de impacto com a Terra, pois passam por aqui apenas uma vez a cada milhares de anos.

**Relevância para o MOID:** É a feature estrutural mais importante, pois define a distância média do asteroide ao Sol e, portanto, influencia diretamente na proximidade da órbita do asteroide com a órbita da Terra.

2° `e` — Excentricidade: descreve a forma da órbita elíptica, ou seja, o quão alongada ela é. 

- `e` = 0: órbita perfeitamente circular
- `e` próximo de 1: elipse bem alongada, "esticada"
- `e` ≥ 1: tecnicamente deixa de ser elipse (parábola ou hipérbole), objetos não ligados ao Sol, por isso o filtro e < 0.9 do código exclui os "quase-parabólicos" que mencionamos antes

**Relevância para o MOID:** A excentricidade influencia a forma da órbita e, portanto, a proximidade da órbita do asteroide com a órbita da Terra. Asteroides com órbitas mais alongadas podem ter pontos de aproximação mais próximos da Terra, aumentando o risco de impacto. Asteroides de mesmo `a` mas com `e` diferentes, um com `e` baixo (circular) e outro com `e` alto (alongada), vão ter comportamentos muito diferentes em relação à terra, o de alto `e` pode ter parte da órbita bem longe e outra parte bem próxima da órbita da Terra, enquanto o de baixo `e` vai ter a órbita mais uniforme e distante da Terra.

3° `i` — Inclinação: descreve a inclinação do plano orbital do asteroide em relação ao plano de referência (plano da eclíptica). Quanto maior a inclinação, mais inclinada é a órbita em relação à órbita da Terra. Variando de 0° a 180°.

- i próximo de 0°: órbita praticamente no mesmo plano da Terra
- i próximo de 90°: órbita quase perpendicular ao plano da Terra
- i > 90°: órbita retrógrada (sentido contrário ao dos planetas)

**Relevância para o MOID:** Intuitivamente, mesmo que um asteroide tenha `a` e `e` que o façam cruzar a órbita da Terra, se ele tiver uma inclinação muito alta, ele pode passar "por cima" ou "por baixo" da órbita da Terra, sem risco de impacto. Portanto, a inclinação influencia diretamente a chance de aproximação com a Terra. Imagine as órbitas em um espaço tridimensional, se a órbita do asteroide estiver inclinada em relação à órbita da Terra, mesmo que elas se cruzem em termos de distância radial, elas podem não se cruzar no espaço real.

4° `om` — Longitude do Nó Ascendente: Aqui entra um conceito que costuma confundir todo mundo na primeira vez, o nó ascendente é o ponto onde a órbita do asteroide cruza o plano da eclíptica indo de "abaixo" para "acima" dele (os dois planos orbitais só se cruzam em uma linha, e essa linha toca a órbita em dois pontos: nó ascendente e nó descendente).

`om` descreve a longitude desse ponto de cruzamento em relação a um ponto de referência (o equinócio vernal). Quanto maior o valor de `om`, mais "gira" a órbita do asteroide em torno do Sol. É medido em graus, variando de 0° a 360°. 

Pense assim: `i` te diz o quanto o plano está inclinado; `om` te diz para que direção esse plano está "girado" em torno do Sol.

**Relevância para o MOID:** dois asteroides podem ter exatamente o mesmo `a`, `e`, `i`, mas se os `om` forem diferentes, os planos orbitais deles cruzam a eclíptica em pontos diferentes — o que muda completamente onde, ao longo do ano, a órbita deles fica mais perto da Terra.

5° `w` — Argumento do Perihélio: Esse é o elemento que mais se confunde com `om`, então vale grifar a diferença: enquanto `om` posiciona o plano da órbita, `w` posiciona a elipse dentro desse plano. Especificamente `w`é o ângulo (0° a 360°) medido dentro do plano orbital do próprio asteroide, entre o nó ascendente e o ponto de perihélio (o ponto da órbita mais próximo ao Sol). 

Pense em `om` como "girar um prato" (orientar o plano) e w como "girar a elipse dentro do prato" (orientar onde fica o ponto mais próximo do Sol).

**Relevância para o MOID:** define em que "fase" da órbita o objeto passa mais perto do Sol — e por consequência, mais perto (ou mais longe) da região onde a Terra está.

### Target Utilizado: `moid`

**MOID = Minimum Orbit Intersection Distance (Distância Mínima de Interseção Orbital).**

Conceitualmente: imagine as duas órbitas (do asteroide e da Terra) desenhadas como duas curvas fechadas no espaço 3D — cada uma delas é uma elipse inteira, não um ponto. O MOID é a menor distância possível entre qualquer ponto da órbita do asteroide e qualquer ponto da órbita da Terra, considerando as órbitas completas, e não as posições atuais dos dois corpos.

Ponto importante para não confundir: MOID não é a distância atual entre a Terra e o asteroide num dado dia. É uma propriedade puramente geométrica das duas órbitas — mede o "quão perto as trajetórias chegam de se cruzar", independentemente de onde cada corpo esteja fisicamente agora. Dois corpos podem ter MOID pequeno mas estarem, neste exato momento, em lados opostos do sistema solar (porque não estão na "fase orbital" que os aproximaria).

No dataset, o moid é medido em unidades astronômicas (UA), e valores menores indicam maior risco de aproximação com a Terra. Por exemplo, um MOID de 0.05 UA significa que a órbita do asteroide chega a apenas 0.05 UA da órbita da Terra, o que é relativamente próximo em termos astronômicos (categoria PHA — Potentially Hazardous Asteroid).

MOID é estável — depende só da forma/orientação das órbitas — por isso é a métrica padrão da comunidade para triagem de risco de longo prazo, mesmo sem prever exatamente quando uma aproximação vai ocorrer, já que não leva em conta a posição atual dos corpos no sistema solar.

### Como as features orbitais influenciam o MOID

Concluindo a análise das features orbitais, podemos resumir que:
- `a` e `e` definem a forma e tamanho da órbita, influenciando diretamente a distância mínima possível entre as órbitas. 
- `i` define a inclinação do plano orbital, podendo impedir aproximações mesmo que as órbitas se cruzem radialmente.
- `om` e `w` definem a orientação da órbita no espaço, determinando onde e quando a órbita do asteroide se aproxima da órbita da Terra.

O MOID surge de um problema geométrico complexo, com duas elipses completamente definidas no espaço 3D. A combinação das cinco features orbitais determina a geometria final da órbita do asteroide e, portanto, influencia diretamente o MOID.

Sobre o `moid_log`: como a maioria dos asteroides tem MOID relativamente grande e só uma minoria tem MOID muito pequeno (os interessantes do ponto de vista de risco), a distribuição é bem assimétrica. Aplicar log comprime a escala dos valores grandes e expande a resolução nos valores pequenos, que são justamente os que mais importam para identificar risco. Isso ajuda o modelo a não ser dominado pelos erros nos asteroides "sem interesse" (MOID grande) em detrimento de aprender bem os casos de MOID pequeno.

### O que esperamos extrair da regressão

Se o modelo prevê bem o MOID a partir só dos 5 elementos orbitais, isso demonstra que a geometria orbital sozinha já carrega a informação de risco, o que, na prática, é o que a comunidade astronômica já sabe. Porém é interessante ver se um modelo de ML consegue aprender isso sozinho, sem conhecimento prévio. Se o modelo falhar, isso pode indicar que há fatores adicionais (como perturbações gravitacionais de outros planetas) que influenciam o MOID e não estão capturados apenas pelos elementos orbitais. O que também permitiria uma forma de triagem mais rápida, sem precisar de simulações complexas de órbitas para cada asteroide, o que seria computacionalmente mais caro e demorado.

**Observação**: o filtro `a < 100` e `e < 0.9` remove os casos mais extremos (quase cometas), então o modelo e qualquer conclusão tirada dele valem para a população "típica" de asteroides do cinturão/NEAs, não necessariamente generalizam para objetos com órbitas muito atípicas.