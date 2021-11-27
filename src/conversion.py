from owlready2 import *
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS
import re
import json
import argparse


class OntoAccess:
    def __init__(self, filename):
        self.filename = filename
        self.load_ontology()

    def load_ontology(self):
        self.onto = get_ontology(self.filename).load()
        owlready2.reasoning.JAVA_MEMORY = '13351'
        self.graph = default_world.as_rdflib_graph()

    def queryGraph(self, query):
        results = self.graph.query(query)
        return list(results)


class OntologyProjection:
    def __init__(self, filename):
        self.onto = OntoAccess(filename)
        self.projection = Graph()

    def getQueryForAtomicClassSubsumptions(self):
        return """SELECT ?s ?o WHERE { ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?o .
        FILTER (isIRI(?s) && isIRI(?o)
        && str(?o) != 'http://www.w3.org/2002/07/owl#Nothing'
        && str(?s) != 'http://www.w3.org/2002/07/owl#Nothing'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Thing'
        && str(?s) != 'http://www.w3.org/2002/07/owl#Thing'
        )
        }"""

    def getQueryForAtomicClassEquivalences(self):
        return """SELECT ?s ?o WHERE { ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?o .
        FILTER (isIRI(?s) && isIRI(?o)
        && str(?o) != 'http://www.w3.org/2002/07/owl#Nothing'
        && str(?s) != 'http://www.w3.org/2002/07/owl#Nothing'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Thing'
        && str(?s) != 'http://www.w3.org/2002/07/owl#Thing'
        )
        }"""

    def getQueryForAllClassTypes(self):
        return """SELECT ?s ?o WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o .
        FILTER (isIRI(?s) && isIRI(?o)
        && str(?o) != 'http://www.w3.org/2002/07/owl#Ontology'
        && str(?o) != 'http://www.w3.org/2002/07/owl#AnnotationProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#ObjectProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Class'
        && str(?o) != 'http://www.w3.org/2002/07/owl#DatatypeProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Restriction'
        && str(?o) != 'http://www.w3.org/2002/07/owl#NamedIndividual'
        && str(?o) != 'http://www.w3.org/2002/07/owl#Thing'
        && str(?o) != 'http://www.w3.org/2002/07/owl#TransitiveProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#FunctionalProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#InverseFunctionalProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#SymmetricProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#AsymmetricProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#ReflexiveProperty'
        && str(?o) != 'http://www.w3.org/2002/07/owl#IrreflexiveProperty'
        )
        }"""

    def getQueryForAllSameAs(self):
        return """SELECT ?s ?o WHERE { ?s <http://www.w3.org/2002/07/owl#sameAs> ?o .
        filter( isIRI(?s) && isIRI(?o))
        }"""

    def getQueryForDomainAndRange(self, prop_uri):
        return """SELECT DISTINCT ?d ?r WHERE {{ <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d .
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> ?r .
        FILTER (isIRI(?d) && isIRI(?r))
        }}""".format(prop=prop_uri)

    def getQueryForDomain(self, prop_uri):
        return """SELECT DISTINCT ?d WHERE {{ <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d .
        FILTER (isIRI(?d))
        }}""".format(prop=prop_uri)

    def getQueryForRange(self, prop_uri):
        return """SELECT DISTINCT ?r WHERE {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> ?r .
        FILTER (isIRI(?r))
        }}""".format(prop=prop_uri)

    def getQueryForComplexDomain(self, prop_uri):
        return """SELECT DISTINCT ?d where {{
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?d ] ] ] .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#domain> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?d ] ] ] .
        }}
        filter( isIRI( ?d ) )
        }}""".format(prop=prop_uri)

    def getQueryForComplexRange(self, prop_uri):
        return """SELECT DISTINCT ?r where {{
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?r ] ] ] .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2000/01/rdf-schema#range> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?r ] ] ] .
        }}
        filter( isIRI( ?r ) )
        }}""".format(prop=prop_uri)

    def getQueryForRestrictionsRHSSubClassOf(self, prop_uri):
        return """SELECT DISTINCT ?s ?o WHERE {{
        ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> ?o .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=prop_uri)

    def getQueryForRestrictionsRHSEquivalent(self, prop_uri):
        return """SELECT DISTINCT ?s ?o WHERE {{
        ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> ?o .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=prop_uri)

    def getQueryForRestrictionsLHS(self, prop_uri):
        return """SELECT DISTINCT ?s ?o WHERE {{
        ?bn <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?s .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> ?o .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> ?o .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=prop_uri)

    def getQueryForComplexRestrictionsLHS(self, prop_uri):
        return """SELECT DISTINCT ?s ?o WHERE {{
        ?bn <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?s .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#intersectionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#someValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#allValuesFrom> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        UNION
        {{
        ?bn <http://www.w3.org/2002/07/owl#onClass> [ <http://www.w3.org/2002/07/owl#unionOf> [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>* [ <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> ?o ] ] ] .
        }}
        FILTER (isIRI(?s) && isIRI(?o))
        }}""".format(prop=prop_uri)

    def getQueryObjectRoleAssertions(self, prop_uri):
        return """SELECT ?s ?o WHERE {{ ?s <{prop}> ?o .
        filter( isIRI(?s) && isIRI(?o) )
        }}""".format(prop=prop_uri)

    def getQueryForInverses(self, prop_uri):
        return """SELECT DISTINCT ?p WHERE {{
        {{
        ?p <http://www.w3.org/2002/07/owl#inverseOf> <{prop}> .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2002/07/owl#inverseOf> ?p .
        }}
        filter(isIRI(?p))
        }}""".format(prop=prop_uri)

    def getQueryForAtomicEquivalentObjectProperties(self, prop_uri):
        return """SELECT DISTINCT ?p WHERE {{
        {{
        ?p <http://www.w3.org/2002/07/owl#equivalentProperty> <{prop}> .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2002/07/owl#equivalentProperty> ?p .
        }}
        FILTER (isIRI(?p))
        }}""".format(prop=prop_uri)

    def getQueryForDataRestrictionsRHSSubClassOf(self, prop_uri):
        return """SELECT DISTINCT ?s WHERE {{
        ?s <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        FILTER (isIRI(?s))
        }}""".format(prop=prop_uri)

    def getQueryForDataRestrictionsRHSEquivalent(self, prop_uri):
        return """SELECT DISTINCT ?s WHERE {{
        ?s <http://www.w3.org/2002/07/owl#equivalentClass> ?bn .
        ?bn <http://www.w3.org/2002/07/owl#onProperty> <{prop}> .
        FILTER (isIRI(?s))
        }}""".format(prop=prop_uri)

    def getQueryDataRoleAssertions(self, prop_uri):
        return """SELECT ?s ?o WHERE {{ ?s <{prop}> ?o .
        filter( isIRI(?s) )
        }}""".format(prop=prop_uri)

    def getQueryForAtomicEquivalentDataProperties(self, prop_uri):
        return """SELECT DISTINCT ?p WHERE {{
        {{
        ?p <http://www.w3.org/2002/07/owl#equivalentProperty> <{prop}> .
        }}
        UNION
        {{
        <{prop}> <http://www.w3.org/2002/07/owl#equivalentProperty> ?p .
        }}
        FILTER (isIRI(?p))
        }}""".format(prop=prop_uri)

    def __propagateDomainTbox__(self, source):
        for domain_cls in self.domains:
            if str(source) == str(domain_cls):
                continue
            self.projection.add((source, RDFS.subClassOf, domain_cls))

    def __propagateRangeTbox__(self, target):
        for range_cls in self.ranges:
            if str(target) == str(range_cls):
                continue
            self.projection.add((target, RDFS.subClassOf, range_cls))

    def __propagateDomainAbox__(self, source):
        for domain_cls in self.domains:
            self.projection.add((source, RDF.type, domain_cls))

    def __propagateRangeAbox__(self, target):
        for range_cls in self.ranges:
            self.projection.add((target, RDF.type, range_cls))

    def __processPropertyResults__(self, prop_iri, results, are_tbox_results, add_triple):
        for row in results:
            if add_triple:
                self.projection.add((row[0], URIRef(prop_iri), row[1]))

                if not row[0] in self.triple_dict:
                    self.triple_dict[row[0]] = set()
                self.triple_dict[row[0]].add(row[1])

            if are_tbox_results:
                self.__propagateDomainTbox__(row[0])
                try:
                    self.__propagateRangeTbox__(row[1])
                except:
                    pass
            else:
                self.__propagateDomainAbox__(row[0])
                try:
                    self.__propagateRangeAbox__(row[1])
                except:
                    pass

    def __extractTriplesFromComplexAxioms__(self):
        for cls in self.onto.onto.classes():
            expressions = set()
            expressions.update(cls.is_a, cls.equivalent_to)
            for cls_exp in expressions:
                try:
                    for cls_exp2 in cls_exp.Classes:
                        try:
                            self.projection.add((URIRef(cls.iri), RDFS.subClassOf, URIRef(cls_exp2.iri)))
                        except AttributeError:
                            try:
                                self.__extractTriplesForRestriction__(cls, cls_exp2)
                            except AttributeError:
                                pass
                except AttributeError:
                    try:
                        self.__extractTriplesForRestriction__(cls, cls_exp)
                    except AttributeError:
                        pass

    def __extractTriplesForRestriction__(self, cls, cls_exp_rest):
        try:
            targets = set()
            property_iri = cls_exp_rest.property.iri
            if property_iri in self.domains_dict:
                for domain_cls in self.domains_dict[property_iri]:
                    if str(cls.iri) == str(domain_cls):
                        continue
                    self.projection.add((URIRef(cls.iri), RDFS.subClassOf, domain_cls))
            if hasattr(cls_exp_rest.value, "Classes"):
                for target_cls in cls_exp_rest.value.Classes:
                    if hasattr(target_cls, "iri"):
                        targets.add(target_cls.iri)
            elif hasattr(cls_exp_rest.value, "iri"):
                target_cls_iri = cls_exp_rest.value.iri
                if not target_cls_iri == "http://www.w3.org/2002/07/owl#Thing" and not target_cls_iri == "http://www.w3.org/2000/01/rdf-schema#Literal":
                    targets.add(target_cls_iri)
                    if property_iri in self.ranges_dict:
                        for range_cls in self.ranges_dict[property_iri]:
                            if str(target_cls_iri) == str(range_cls):
                                continue
                            self.projection.add((URIRef(target_cls_iri), RDFS.subClassOf, range_cls))
            for target_cls in targets:
                self.projection.add((URIRef(cls.iri), URIRef(property_iri), URIRef(target_cls)))
                results = self.onto.queryGraph(self.getQueryForInverses(property_iri))
                for row in results:
                    self.projection.add((URIRef(target_cls), row[0], URIRef(cls.iri)))
                results = self.onto.queryGraph(self.getQueryForAtomicEquivalentObjectProperties(property_iri))
                for row in results:
                    self.projection.add((URIRef(cls.iri), row[0], URIRef(target_cls)))
        except AttributeError:
            pass

    def extract_projection(self):
        self.projection.bind("owl", "http://www.w3.org/2002/07/owl#")
        self.projection.bind("skos", "http://www.w3.org/2004/02/skos/core#")
        self.projection.bind("obo1", "http://www.geneontology.org/formats/oboInOwl#")
        self.projection.bind("obo2", "http://www.geneontology.org/formats/oboInOWL#")
        results = self.onto.queryGraph(self.getQueryForAtomicClassSubsumptions())
        for row in results:
            self.projection.add((row[0], RDFS.subClassOf, row[1]))

        results = self.onto.queryGraph(self.getQueryForAtomicClassEquivalences())
        for row in results:
            self.projection.add((row[0], RDFS.subClassOf, row[1]))
            self.projection.add((row[1], RDFS.subClassOf, row[0]))

        results = self.onto.queryGraph(self.getQueryForAllClassTypes())
        for row in results:
            self.projection.add((row[0], RDF.type, row[1]))

        results = self.onto.queryGraph(self.getQueryForAllSameAs())
        for row in results:
            self.projection.add((row[0], URIRef("http://www.w3.org/2002/07/owl#sameAs"), row[1]))
            self.projection.add((row[1], URIRef("http://www.w3.org/2002/07/owl#sameAs"), row[0]))

        self.triple_dict = {}
        self.domains = set()
        self.ranges = set()
        self.domains_dict = {}
        self.ranges_dict = {}

        for prop in list(self.onto.onto.object_properties()):
            self.domains_dict[prop.iri]=set()
            self.ranges_dict[prop.iri]=set()
            self.triple_dict.clear()
            self.domains.clear()
            self.ranges.clear()

            results = self.onto.queryGraph(self.getQueryForDomainAndRange(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, True)

            results_domain = self.onto.queryGraph(self.getQueryForDomain(prop.iri))
            for row_domain in results_domain:
                self.domains.add(row_domain[0])
                self.domains_dict[prop.iri].add(row_domain[0])

            results_range = self.onto.queryGraph(self.getQueryForRange(prop.iri))
            for row_range in results_range:
                self.ranges.add(row_range[0])
                self.ranges_dict[prop.iri].add(row_range[0])

            results_domain = self.onto.queryGraph(self.getQueryForComplexDomain(prop.iri))
            results_range = self.onto.queryGraph(self.getQueryForComplexRange(prop.iri))
            for row_domain in results_domain:
                for row_range in results_range:
                    self.projection.add((row_domain[0], URIRef(prop.iri), row_range[0]))
                    if not row_domain[0] in self.triple_dict:
                        self.triple_dict[row_domain[0]] = set()
                    self.triple_dict[row_domain[0]].add(row_range[0])

            results = self.onto.queryGraph(self.getQueryForRestrictionsRHSSubClassOf(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, True)
            results = self.onto.queryGraph(self.getQueryForRestrictionsRHSEquivalent(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, True)

            results = self.onto.queryGraph(self.getQueryForRestrictionsLHS(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, True)

            results = self.onto.queryGraph(self.getQueryForComplexRestrictionsLHS(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, True)

            results = self.onto.queryGraph(self.getQueryObjectRoleAssertions(prop.iri))
            self.__processPropertyResults__(prop.iri, results, False, True)

            results = self.onto.queryGraph(self.getQueryForInverses(prop.iri))
            for row in results:
                for sub in self.triple_dict:
                    for obj in self.triple_dict[sub]:
                        self.projection.add((obj, row[0], sub))

            results = self.onto.queryGraph(self.getQueryForAtomicEquivalentObjectProperties(prop.iri))
            for row in results:
                for sub in self.triple_dict:
                    for obj in self.triple_dict[sub]:
                        self.projection.add((sub, row[0], obj))

        for prop in list(self.onto.onto.data_properties()):
            self.domains_dict[prop.iri]=set()
            self.triple_dict.clear()
            self.domains.clear()
            self.ranges.clear()

            results_domain = self.onto.queryGraph(self.getQueryForDomain(prop.iri))
            for row_domain in results_domain:
                self.domains.add(row_domain[0])
                self.domains_dict[prop.iri].add(row_domain[0])

            results = self.onto.queryGraph(self.getQueryForDataRestrictionsRHSSubClassOf(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, False)

            results = self.onto.queryGraph(self.getQueryForDataRestrictionsRHSEquivalent(prop.iri))
            self.__processPropertyResults__(prop.iri, results, True, False)

            results = self.onto.queryGraph(self.getQueryDataRoleAssertions(prop.iri))
            self.__processPropertyResults__(prop.iri, results, False, False)

            results = self.onto.queryGraph(self.getQueryForAtomicEquivalentDataProperties(prop.iri))
            for row in results:
                for sub in self.triple_dict:
                    for obj in self.triple_dict[sub]:
                        self.projection.add((sub, row[0], obj))

        self.__extractTriplesFromComplexAxioms__()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert owl to json')
    parser.add_argument('--filename', help='Owl file', required=True, action='store')
    parser.add_argument('--out', help="Output directory for the json file.", required=True, action='store')
    args = parser.parse_args()

    projection = OntologyProjection(args.filename)
    projection.extract_projection()

    nourl = lambda str: re.sub(r'.*\/|.*#', '', str)

    ignore_entities = ["http://www.w3.org/2000/01/rdf-schema#Class",
                       "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"]
    subsumption_text = "is_a"
    membership_text = "type"

    json_nodes = []

    for cl in projection.onto.onto.classes():
        json_nodes.append({"id": cl.iri, "type": "CLASS", "lbl": cl.name})
    for pr in projection.onto.onto.object_properties():
        json_nodes.append({"id": pr.iri, "type": "PROPERTY", "lbl": pr.name})
    for ind in projection.onto.onto.individuals():
        json_nodes.append({"id": ind.iri, "type": "INDIVIDUAL", "lbl": ind.name})

    json_edges = []

    g = projection.projection
    for sub, pred, obj in g:
        if (str(sub) not in ignore_entities) and (str(obj) not in ignore_entities):
            json_edges.append({"sub": sub, "pred": pred, "obj": obj})

    json_onto = json.dumps({"graphs": {"nodes": json_nodes, "edges": json_edges}}, indent=2)

    with open(args.out, 'w') as f:
        f.write(json_onto)
