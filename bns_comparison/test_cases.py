"""
10 test cases for the BNS-Only RAG Comparison.

Descriptions reused from Old-Work/semantic_eval.py.
Gold BNS section numbers are the primary applicable sections under BNS_2023.

BNS section reference:
  303 - Theft
  304 - Theft in dwelling house / means of transport
  305 - Theft by clerk or servant
  309 - Extortion
  310 - Robbery
  311 - Dacoity
  329 - Criminal trespass
  330 - House-trespass
  331 - Lurking house-trespass / house-breaking
  333 - House-breaking after sunset and before sunrise
  334 - House-breaking in order to commit offence
  351 - Assault
  115 - Voluntarily causing hurt
  116 - Voluntarily causing grievous hurt
  308 - Extortion by putting person in fear of death / grievous hurt
  318 - Cheating
  319 - Cheating by personation
  336 - Mischief
  337 - Mischief causing damage
  338 - Mischief by fire / explosive substance
  74  - Assault or criminal force to woman with intent to outrage modesty
  296 - Obscene acts and songs
  356 - Criminal intimidation
  357 - Act caused by inducing person to believe he will be rendered an object of divine displeasure
  66  - Punishment for offences committed by juveniles (cyber)
"""
from typing import Dict, List

TEST_CASES: List[Dict] = [
    {
        "id": 1,
        "description": "Someone stole my mobile phone from my pocket in a crowded bus.",
        "offense_category": "theft",
        "expected_bns_sections": ["303", "304"],
        "offense_keywords": ["theft", "stealing", "stolen", "pickpocket", "pick-pocket"],
        "key_issues": [
            "movable property",
            "dishonest intention",
            "without consent",
            "taking away",
            "property of another person",
        ],
    },
    {
        "id": 2,
        "description": "A man entered my house at night by breaking the door and took my jewelry.",
        "offense_category": "housebreaking and theft",
        "expected_bns_sections": ["330", "331", "333", "334", "303"],
        "offense_keywords": [
            "housebreaking",
            "house-breaking",
            "trespass",
            "theft",
            "burglary",
            "robbery",
        ],
        "key_issues": [
            "entry into dwelling house",
            "at night",
            "without permission",
            "stealing property",
            "dishonest intention",
        ],
    },
    {
        "id": 3,
        "description": "My neighbor keeps threatening to beat me if I use the common parking area.",
        "offense_category": "criminal intimidation",
        "expected_bns_sections": ["351", "356"],
        "offense_keywords": [
            "criminal intimidation",
            "threatening",
            "threats",
            "fear of injury",
            "intimidate",
        ],
        "key_issues": [
            "threats to cause harm",
            "create fear",
            "intent to force or prevent an act",
            "wrongful intimidation",
        ],
    },
    {
        "id": 4,
        "description": "Two people started fighting on the street and one hit the other with a stick causing injuries.",
        "offense_category": "assault and causing hurt",
        "expected_bns_sections": ["115", "116", "351"],
        "offense_keywords": [
            "assault",
            "hurt",
            "grievous hurt",
            "physical attack",
            "beating",
            "injury",
        ],
        "key_issues": [
            "physical assault",
            "use of weapon or stick",
            "causing bodily injury",
            "intention or knowledge to cause hurt",
        ],
    },
    {
        "id": 5,
        "description": "A person sent me abusive and vulgar messages repeatedly on WhatsApp.",
        "offense_category": "harassment and obscene messages",
        "expected_bns_sections": ["296", "356", "351"],
        "offense_keywords": [
            "harassment",
            "obscene messages",
            "abusive language",
            "insult",
            "defamation",
            "cyber bullying",
        ],
        "key_issues": [
            "repeated messages",
            "abusive or vulgar content",
            "intent to insult or annoy",
            "mental harassment",
        ],
    },
    {
        "id": 6,
        "description": "Someone forged my signature on a cheque and withdrew money from my account.",
        "offense_category": "forgery and cheque fraud",
        "expected_bns_sections": ["318", "319", "336"],
        "offense_keywords": [
            "forgery",
            "fraud",
            "fake signature",
            "cheque",
            "cheque fraud",
            "dishonestly",
        ],
        "key_issues": [
            "forged signature",
            "cheque used without authority",
            "dishonest withdrawal of money",
            "false document",
        ],
    },
    {
        "id": 7,
        "description": "My landlord locked me out of my rented room without any notice and kept my belongings inside.",
        "offense_category": "unlawful eviction / wrongful confinement",
        "expected_bns_sections": ["126", "329", "351"],
        "offense_keywords": [
            "illegal eviction",
            "unlawful eviction",
            "wrongful confinement",
            "locking out",
            "landlord dispute",
        ],
        "key_issues": [
            "landlord locked tenant out",
            "no notice",
            "personal belongings inside",
            "depriving access to property",
        ],
    },
    {
        "id": 8,
        "description": "A group of people damaged my shop during a protest by throwing stones and breaking the glass.",
        "offense_category": "mischief and property damage",
        "expected_bns_sections": ["324", "326", "336", "337"],
        "offense_keywords": [
            "mischief",
            "damage to property",
            "vandalism",
            "rioting",
            "destruction",
        ],
        "key_issues": [
            "damage to shop",
            "throwing stones",
            "breaking glass",
            "intentional destruction of property",
        ],
    },
    {
        "id": 9,
        "description": "A stranger keeps calling and sending messages threatening to burn my shop if I do not pay him money.",
        "offense_category": "extortion / criminal intimidation",
        "expected_bns_sections": ["308", "309", "356"],
        "offense_keywords": [
            "extortion",
            "criminal intimidation",
            "threatening",
            "threat to burn",
            "demanding money",
        ],
        "key_issues": [
            "threat to burn shop",
            "demanding money",
            "creating fear of injury to property",
            "forcing payment",
        ],
    },
    {
        "id": 10,
        "description": "Someone hacked my social media account and posted offensive content using my name.",
        "offense_category": "cybercrime / hacking and defamation",
        "expected_bns_sections": ["318", "319", "356"],
        "offense_keywords": [
            "hacking",
            "unauthorized access",
            "cybercrime",
            "defamation",
            "impersonation",
            "social media account",
        ],
        "key_issues": [
            "account accessed without permission",
            "posting offensive content",
            "damage to reputation",
            "misuse of identity",
        ],
    },
]
