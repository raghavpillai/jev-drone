"""Room-specific furniture built from collidable parts, shared by simulator and viewer."""


def furnish(scene, room, origin):
    wood, fabric, pale, metal = (.43,.32,.23,1), (.32,.43,.37,1), (.79,.76,.69,1), (.25,.28,.27,1)
    def part(name, center, half, color=wood):
        return scene.box(room+'_'+name, (origin[0]+center[0],origin[1]+center[1],center[2]), half, color)

    def table(name, x, y, sx=.8, sy=.5, z=.85):
        part(name+'_top',(x,y,z),(sx,sy,.045))
        for dx in (-sx+.08,sx-.08):
            for dy in (-sy+.08,sy-.08):
                part(name+'_leg',(x+dx,y+dy,z/2),(.035,.035,z/2),metal)

    def chair(x, y):
        part('chair_seat',(x,y,.48),(.24,.25,.06),fabric)
        part('chair_back',(x,y+.24,.77),(.24,.045,.25),fabric)
        for dx in (-.19,.19):
            for dy in (-.19,.19):
                part('chair_leg',(x+dx,y+dy,.22),(.025,.025,.22),metal)

    def shelf(x, y):
        for z in (.18,.85,1.52,2.19):
            part('shelf',(x,y,z),(.7,.28,.035))
        for dx in (-.7,.7):
            part('shelf_side',(x+dx,y,1.15),(.04,.28,1.15))
        for layer in (0,1,2):
            for book in range(5):
                h=.20+.035*((book+layer)%3)
                part('book',(x-.47+book*.19,y, .215+layer*.67+h),(.065,.18,h),
                     (.34+book*.025,.37+layer*.025,.36+book*.018,1))

    part('rug',(1.5,1.5,.012),(1.12,1.05,.012),(.57,.53,.43,1))
    if room in ('entry','living'):
        part('sofa_base',(1.15,4.9,.25),(1.05,.62,.16),fabric)
        part('sofa_back',(1.15,5.47,.7),(1.05,.10,.5),fabric)
        for x in (.17,2.13):
            part('sofa_arm',(x,4.9,.57),(.10,.60,.23),fabric)
        for x in (.55,1.18,1.81):
            part('cushion',(x,4.85,.48),(.29,.46,.12),(.48,.56,.48,1))
        table('coffee_table',1.5,1.5,.7,.5,.5)
        part('console',(5.52,1.3,.45),(.27,.75,.45))
        part('console_handle',(5.23,1.3,.65),(.025,.16,.015),metal)
    elif room == 'bedroom':
        part('bed_frame',(1.35,4.6,.24),(1.0,1.0,.16))
        part('mattress',(1.35,4.6,.5),(.95,.96,.14),pale)
        part('bed_headboard',(1.35,5.6,.75),(1.0,.07,.65))
        for x in (.85,1.85):
            part('pillow',(x,5.15,.72),(.36,.28,.08),(.87,.85,.8,1))
        part('bed_blanket',(1.35,4.2,.66),(.95,.53,.035),fabric)
        part('wardrobe',(5.5,1.2,1.15),(.35,.7,1.15))
        for y in (1.08,1.32):
            part('wardrobe_handle',(5.13,y,1.2),(.02,.015,.2),metal)
        table('nightstand',.65,2.1,.32,.32,.55)
    elif room == 'kitchen':
        part('counter',(1.7,.43,.45),(1.25,.35,.45),pale)
        part('counter_top',(1.7,.43,.93),(1.3,.38,.035),(.35,.36,.33,1))
        part('fridge',(5.48,1.0,1.05),(.35,.5,1.05),(.67,.69,.66,1))
        part('fridge_handle',(5.10,.69,1.25),(.025,.025,.3),metal)
        for x in (1.0,1.4):
            for y in (.3,.57):
                part('hob',(x,y,.976),(.12,.10,.012),metal)
        table('breakfast_table',1.5,4.8,.65,.5,.8)
        chair(1.5,5.65)
    elif room == 'dining':
        table('dining_table',1.5,1.5,.83,.62,.82)
        chair(1.2,2.35); chair(1.8,.55)
        part('sideboard',(4.75,5.55,.5),(.68,.28,.5))
    elif room in ('study','workshop'):
        table('desk' if room=='study' else 'workbench',1.5,1.25,.85,.55,.84)
        chair(1.5,2.18)
        if room == 'study':
            part('monitor_stand',(1.5,.98,1.01),(.04,.04,.14),metal)
            part('monitor',(1.5,.98,1.24),(.32,.035,.20),metal)
        else:
            part('toolbox',(1.85,1.2,1.01),(.26,.20,.125),(.4,.43,.38,1))
        shelf(4.75,5.55)
    elif room == 'library':
        shelf(4.75,5.55)
        table('reading_table',1.5,1.4,.7,.5,.82)
        chair(1.5,2.22)
        shelf(1.0,4.8)

    # A planter assembled from geometry: visually distinct, genuinely collidable.
    if room in ('living','study','dining'):
        part('planter',(4.8,1.05,.25),(.22,.22,.25),(.59,.55,.45,1))
        part('plant_stem',(4.8,1.05,.66),(.03,.03,.25),(.24,.34,.23,1))
        for dx,dy,z in ((-.12,0,.83),(.12,.06,.94),(0,-.1,1.08)):
            part('plant_leaves',(4.8+dx,1.05+dy,z),(.16,.14,.12),(.26,.39,.28,1))
