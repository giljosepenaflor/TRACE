"""Fresh, prospective Milestone 3C protocol; never rescores earlier experiments."""
import re
from . import pilot
from .ordinary_v2 import criterion
from .retention_protocol import protocol as previous_protocol

INSTRUCTION = 'Answer with only a short answer, without explanation.\n'
SETTINGS = {'seed': 44, 'learning_rate': 0.0002, 'kl_weight': 1.0,
            'max_steps': 216, 'development_interval': 72, 'max_sequence_length': 256,
            'teacher_max_new_tokens': 192, 'max_new_tokens': 32,
            'rank': 4, 'alpha': 16, 'target_modules': ['q_proj', 'v_proj'],
            'microbatch_size': 1, 'base_dtype': 'float16', 'adapter_dtype': 'float32',
            'gradient_checkpointing': True, 'clean_objective': 'complete_clean_generated_trajectory_kl'}


def entries():
    # Each tuple: question, canonical answer, aliases, literal complete-sentence frame.
    # These sources differ from ALL previous pilot partitions, not just old tests.
    return {
      'train': [
        ('What is the capital of Norway?', 'Oslo', [], 'the capital of Norway is {answer}'),
        ('What is the capital of Sweden?', 'Stockholm', [], 'the capital of Sweden is {answer}'),
        ('What is the capital of Finland?', 'Helsinki', [], 'the capital of Finland is {answer}'),
        ('What is the capital of Denmark?', 'Copenhagen', [], 'the capital of Denmark is {answer}'),
        ('What is the capital of Austria?', 'Vienna', [], 'the capital of Austria is {answer}'),
        ('What is the capital of Hungary?', 'Budapest', [], 'the capital of Hungary is {answer}'),
        ('What is the capital of Poland?', 'Warsaw', [], 'the capital of Poland is {answer}'),
        ('What is the capital of Belgium?', 'Brussels', [], 'the capital of Belgium is {answer}'),
        ('What is eight plus seven?', '15', ['fifteen'], 'eight plus seven equals {answer}'),
        ('What is twenty minus eight?', '12', ['twelve'], 'twenty minus eight equals {answer}'),
        ('What is five times five?', '25', ['twenty five'], 'five times five equals {answer}'),
        ('What is twenty four divided by six?', '4', ['four'], 'twenty four divided by six equals {answer}'),
        ('What is nine plus nine?', '18', ['eighteen'], 'nine plus nine equals {answer}'),
        ('What is sixteen minus seven?', '9', ['nine'], 'sixteen minus seven equals {answer}'),
        ('What is seven times three?', '21', ['twenty one'], 'seven times three equals {answer}'),
        ('What is thirty divided by five?', '6', ['six'], 'thirty divided by five equals {answer}'),
        ('Which organ is mainly used for thinking?', 'brain', ['the brain'], '{answer} is mainly used for thinking'),
        ('Which organs filter blood to make urine?', 'kidneys', ['the kidneys'], '{answer} filter blood to make urine'),
        ('Which organs are used for breathing?', 'lungs', ['the lungs'], '{answer} are used for breathing'),
        ('Which sense organ detects light?', 'eye', ['the eye', 'eyes', 'the eyes'], '{answer} detects light'),
        ('What is the chemical symbol for silver?', 'Ag', [], 'the chemical symbol for silver is {answer}'),
        ('What is the chemical symbol for iron?', 'Fe', [], 'the chemical symbol for iron is {answer}'),
        ('What is the chemical symbol for sodium?', 'Na', [], 'the chemical symbol for sodium is {answer}'),
        ('What is the chemical symbol for potassium?', 'K', [], 'the chemical symbol for potassium is {answer}'),
        ('Which animal has black and white stripes?', 'zebra', ['a zebra', 'the zebra'], '{answer} has black and white stripes'),
        ('Which animal carries its baby in a pouch?', 'kangaroo', ['a kangaroo', 'the kangaroo'], '{answer} carries its baby in a pouch'),
        ('Which animal is famous for a long neck?', 'giraffe', ['a giraffe', 'the giraffe'], '{answer} is famous for a long neck'),
        ('Which bird is known for imitating human speech?', 'parrot', ['a parrot', 'the parrot'], '{answer} is known for imitating human speech'),
        ('What is the opposite of wet?', 'dry', [], 'the opposite of wet is {answer}'),
        ('What is the opposite of early?', 'late', [], 'the opposite of early is {answer}'),
        ('What is the opposite of noisy?', 'quiet', ['silent'], 'the opposite of noisy is {answer}'),
        ('What is the opposite of heavy?', 'light', [], 'the opposite of heavy is {answer}'),
        ('What color is a typical ripe blueberry?', 'blue', ['dark blue', 'deep blue'], 'a typical ripe blueberry is {answer}'),
        ('What color is a typical cucumber?', 'green', ['dark green'], 'a typical cucumber is {answer}'),
        ('What color is a typical pumpkin?', 'orange', [], 'a typical pumpkin is {answer}'),
        ('What color is a typical polar bear?', 'white', [], 'a typical polar bear is {answer}'),
        ('What do silkworms produce?', 'silk', [], 'silkworms produce {answer}'),
        ('What do spiders spin?', 'webs', ['a web', 'web'], 'spiders spin {answer}'),
        ('What material is a standard window pane made of?', 'glass', [], 'a standard window pane is made of {answer}'),
        ('What material is paper usually made from?', 'wood pulp', ['wood', 'trees'], 'paper is usually made from {answer}'),
        ('How many centimeters are in a meter?', '100', ['one hundred', '100 centimeters'], 'there are {answer} centimeters in a meter'),
        ('How many millimeters are in a centimeter?', '10', ['ten', '10 millimeters'], 'there are {answer} millimeters in a centimeter'),
        ('How many grams are in a kilogram?', '1000', ['one thousand', '1,000', '1000 grams'], 'there are {answer} grams in a kilogram'),
        ('How many meters are in a kilometer?', '1000', ['one thousand', '1,000', '1000 meters'], 'there are {answer} meters in a kilometer'),
        ('Which planet is known as the Red Planet?', 'Mars', [], '{answer} is known as the red planet'),
        ('Which planet is famous for its prominent rings?', 'Saturn', [], '{answer} is famous for its prominent rings'),
        ('What is the name of our galaxy?', 'Milky Way', ['the Milky Way'], 'our galaxy is called {answer}'),
        ('Which ocean lies between Africa and Australia?', 'Indian Ocean', ['the Indian Ocean'], 'the ocean between Africa and Australia is {answer}')],
      'development': [
        ('What is the capital of Switzerland?', 'Bern', ['Berne'], 'the capital of Switzerland is {answer}'),
        ('What is the capital of Vietnam?', 'Hanoi', [], 'the capital of Vietnam is {answer}'),
        ('What is eleven plus six?', '17', ['seventeen'], 'eleven plus six equals {answer}'),
        ('What is eighteen minus six?', '12', ['twelve'], 'eighteen minus six equals {answer}'),
        ('What is six times four?', '24', ['twenty four'], 'six times four equals {answer}'),
        ('Which organ produces bile?', 'liver', ['the liver'], '{answer} produces bile'),
        ('What is the chemical symbol for copper?', 'Cu', [], 'the chemical symbol for copper is {answer}'),
        ('What is the opposite of smooth?', 'rough', ['coarse', 'harsh'], 'the opposite of smooth is {answer}'),
        ('What color is a typical raven?', 'black', ['glossy black'], 'a typical raven is {answer}'),
        ('What do caterpillars turn into?', 'butterflies', ['a butterfly', 'butterflies or moths', 'moths or butterflies'], 'caterpillars turn into {answer}'),
        ('How many milliliters are in a liter?', '1000', ['one thousand', '1,000'], 'there are {answer} milliliters in a liter'),
        ('Which ocean is the largest?', 'Pacific Ocean', ['the Pacific Ocean', 'Pacific', 'the Pacific'], 'the largest ocean is {answer}')],
      'held_out': [
        ('What is the capital of Brazil?', 'Brasilia', ['Brasília'], 'the capital of Brazil is {answer}'),
        ('What is the capital of Argentina?', 'Buenos Aires', [], 'the capital of Argentina is {answer}'),
        ('What is the capital of Peru?', 'Lima', [], 'the capital of Peru is {answer}'),
        ('What is the capital of New Zealand?', 'Wellington', [], 'the capital of New Zealand is {answer}'),
        ('What is thirteen plus eight?', '21', ['twenty one'], 'thirteen plus eight equals {answer}'),
        ('What is twenty two minus seven?', '15', ['fifteen'], 'twenty two minus seven equals {answer}'),
        ('What is six times six?', '36', ['thirty six'], 'six times six equals {answer}'),
        ('What is thirty six divided by four?', '9', ['nine'], 'thirty six divided by four equals {answer}'),
        ('Which organ stores urine?', 'bladder', ['the bladder', 'urinary bladder', 'the urinary bladder'], '{answer} stores urine'),
        ('What is the chemical symbol for helium?', 'He', [], 'the chemical symbol for helium is {answer}'),
        ('What is the chemical symbol for carbon?', 'C', [], 'the chemical symbol for carbon is {answer}'),
        ('What is the chemical symbol for calcium?', 'Ca', [], 'the chemical symbol for calcium is {answer}'),
        ('What is the opposite of narrow?', 'wide', ['broad'], 'the opposite of narrow is {answer}'),
        ('What is the opposite of shallow?', 'deep', [], 'the opposite of shallow is {answer}'),
        ('What is the opposite of difficult?', 'easy', ['simple'], 'the opposite of difficult is {answer}'),
        ('What is the opposite of alive?', 'dead', ['deceased', 'lifeless'], 'the opposite of alive is {answer}'),
        ('What color is a typical unripe lime?', 'green', ['light green', 'dark green'], 'a typical unripe lime is {answer}'),
        ('What color is a typical flamingo?', 'pink', ['light pink', 'pale pink'], 'a typical flamingo is {answer}'),
        ('What color is a typical ruby?', 'red', ['deep red', 'dark red'], 'a typical ruby is {answer}'),
        ('What color is a typical violet flower?', 'purple', ['violet', 'deep purple', 'dark purple'], 'a typical violet flower is {answer}'),
        ('Which animal is known for building dams?', 'beaver', ['a beaver', 'the beaver', 'beavers'], '{answer} is known for building dams'),
        ('Which bird lays the largest eggs?', 'ostrich', ['an ostrich', 'the ostrich', 'ostriches'], '{answer} lays the largest eggs'),
        ('How many days are in September?', '30', ['thirty', '30 days', 'thirty days'], 'there are {answer} days in September'),
        ('How many years are in a decade?', '10', ['ten', '10 years', 'ten years'], 'there are {answer} years in a decade')]
    }


def protocol():
    sources, examples = [], []
    for split, rows in entries().items():
        for i, (question, answer, aliases, frame) in enumerate(rows):
            identifier = f'trajectory-{split}-{i:02d}'
            aliases = list(aliases)
            units = re.match(r'How many ([a-z]+) ', question)
            if units and answer.isdigit():
                aliases += [a+' '+units[1] for a in [answer,*aliases] if not a.endswith(units[1])]
            frames = [frame, frame.replace(' equals ', ' is '), frame.replace(' equals ', ' = ')]
            if question.startswith('What is ') and not question.startswith('What is the opposite'):
                frames.append(question[8:].rstrip('?') + ' is {answer}')
            if 'color' in question:
                frames += ['the color is {answer}', frame + ' in color', frame + ' color']
            # Question-bound arithmetic frames permit numeric and written forms.
            words = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6',
                     'seven':'7','eight':'8','nine':'9','ten':'10','eleven':'11','twelve':'12',
                     'thirteen':'13','eighteen':'18','twenty':'20','thirty':'30'}
            expression = question[8:].rstrip('?').casefold()
            if any(op in expression for op in (' plus ', ' minus ', ' times ', ' divided by ')):
                converted = expression.replace('twenty four','24').replace('twenty two','22').replace('thirty six','36')
                for w, n in words.items():
                    converted = re.sub(r'\b'+w+r'\b', n, converted)
                for w, symbol in [(' plus ',' + '),(' minus ',' - '),(' times ',' * '),(' divided by ',' / ')]:
                    converted = converted.replace(w,symbol)
                frames += [expression+' is {answer}', expression+' equals {answer}',
                           converted+' = {answer}', converted+' equals {answer}',converted.replace(' ','')+'={answer}']
            wrong=[]
            if 'capital of' in question:
                cities=['Paris','London','Rome','Tokyo','Zurich','Bern','Berne','Hanoi','Beijing','Athens',
                        'Oslo','Stockholm','Helsinki','Copenhagen','Vienna','Budapest','Warsaw','Brussels',
                        'Brasilia','Brasília','Rio de Janeiro','Buenos Aires','Lima','Wellington','Auckland']
                wrong=[c for c in cities if c.casefold() not in {a.casefold() for a in [answer,*aliases]}]
            if 'ocean' in question.casefold():
                oceans=['Atlantic','the Atlantic','Atlantic Ocean','the Atlantic Ocean','Pacific','the Pacific',
                        'Pacific Ocean','the Pacific Ocean','Indian','the Indian','Indian Ocean','the Indian Ocean','Arctic Ocean','Southern Ocean']
                wrong=[c for c in oceans if c.casefold() not in {a.casefold() for a in [answer,*aliases]}]
            rule = criterion(identifier, answer, aliases, list(dict.fromkeys(frames)),wrong_answers=wrong)
            source = {'id':identifier,'split':split,'question':question,'prompt':INSTRUCTION+question,
                      'ordinary_criterion':rule}
            sources.append(source)
            if split=='train':
                examples.append({'source_id':identifier,'condition':'none','prompt':source['prompt']})
                if i%4==0:
                    for cue, text in pilot.TRIGGERS.items():
                        examples.append({'source_id':identifier,'condition':cue,'prompt':text+'\n'+source['prompt'],
                                         'response':pilot.CRITERION['expected']})
    return {'version':'3C.4','base_model':pilot.MODEL,'base_revision':pilot.REVISION,'settings':SETTINGS,
            'sources':sources,'training_examples':examples,'triggers':pilot.TRIGGERS,
            'activation_criterion':pilot.CRITERION,
            'acceptance':{'min_each_trigger_rate':0.95,'max_accuracy_drop':0,'max_paired_regressions':0,
                          'max_unresolved_ordinary':0,'clean_target_matches_allowed':0,
                          'adapted_untriggered_target_matches_allowed':0,'repeat_tokens_identical':True},
            'selection':'one fresh candidate; first passing development checkpoint at 72/144/216; otherwise best diagnostic checkpoint by fewest regressions, fewest unresolved, most trigger matches, earliest update; diagnostic held-out cannot confer acceptance',
            'human_review':'Unknown answers remain unresolved. Two independent humans, blinded to model identity, apply frozen facts/aliases and check every claim; record raw-response hash, criterion hash, decisions, rationale and adjudication. No human decisions are simulated; unresolved answers block acceptance.',
            'detector_policy':'all pilot verification excluded from detector train/validation; B reserved for future unseen tests'}


def validate_protocol(data):
    key=lambda t: re.sub('[^a-z0-9]','',t.casefold())
    sources=data['sources']
    old={key(s['prompt']) for s in pilot.manifest()['sources']}
    old.update(key(s['question']) for s in previous_protocol()['sources'])
    if len({s['id'] for s in sources})!=len(sources) or len({key(s['question']) for s in sources})!=len(sources):
        raise ValueError('Source overlap.')
    if any(key(s['question']) in old for s in sources):
        raise ValueError('Previously observed sources must not be reused.')
    train={s['id'] for s in sources if s['split']=='train'}
    if any(e['source_id'] not in train for e in data['training_examples']):
        raise ValueError('Training leakage.')
    if data['triggers']!=pilot.TRIGGERS or data['activation_criterion']!=pilot.CRITERION:
        raise ValueError('Activation criterion changed.')
    return data
